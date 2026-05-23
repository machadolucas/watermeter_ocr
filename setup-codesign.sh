#!/usr/bin/env bash
# One-time setup: generate a self-signed code-signing identity and import it
# into the login keychain so install.sh can sign the watermeter binary
# without prompting.
#
# Why this exists
# ---------------
# macOS TCC keys the Local Network privacy grant on the binary's code
# signing identity. Without a stable signature, every rebuild produces a new
# cdhash, TCC sees a new identity, and the next launch pops a "allow access
# to devices on your local network" modal on the server's screen. By signing
# every build with the same self-signed identity, the grant the user
# approves once persists across all future rebuilds.
#
# Run this ONCE per machine. Subsequent runs are no-ops.
#
# What it does
# ------------
#   1. Generates a 100-year RSA-2048 code-signing cert (CN="Macserver Code Signing").
#   2. Packs cert+key into a PKCS12 with legacy ciphers (SHA1+3DES) so
#      macOS's `security import` accepts it — OpenSSL 3.x's default
#      PBES2+AES is rejected silently.
#   3. Imports the p12 into ~/Library/Keychains/login.keychain-db, marking the
#      private key as accessible to /usr/bin/codesign without a GUI prompt.
#   4. Adds the cert as a code-signing trust root in the user trust domain,
#      so codesign won't prompt with "not trusted" warnings either.
#
# The private key never leaves the keychain — it's protected by the user's
# login. The cert is self-signed and trusted only on this machine, which is
# exactly what we want: TCC anchors on this identity, but nothing else does.

set -euo pipefail

IDENTITY_CN="Macserver Code Signing"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning "$KEYCHAIN" 2>/dev/null | grep -q "\"$IDENTITY_CN\""; then
  echo "[setup-codesign] identity \"$IDENTITY_CN\" already in keychain — nothing to do."
  exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Apple's /usr/bin/openssl (LibreSSL) is pinned deliberately: Homebrew's
# OpenSSL 3.x writes PKCS12 with PBES2+AES, which `security import` rejects.
echo "[setup-codesign] generating key + self-signed cert"
/usr/bin/openssl genrsa -out "$WORK/signing.key" 2048 2>/dev/null
/usr/bin/openssl req -new -x509 \
  -key "$WORK/signing.key" \
  -out "$WORK/signing.crt" \
  -days 36500 \
  -subj "/CN=$IDENTITY_CN/O=machadolucas/C=FI" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=critical,codeSigning" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "subjectKeyIdentifier=hash" 2>/dev/null

# PKCS12 needs a non-empty password for macOS's `security import` MAC check.
# The password only protects the in-transit p12 in /tmp (which we shred); the
# key in the keychain is protected by the user's login.
P12_PASS="$(/usr/bin/openssl rand -hex 16)"
/usr/bin/openssl pkcs12 -export \
  -out "$WORK/signing.p12" \
  -inkey "$WORK/signing.key" \
  -in "$WORK/signing.crt" \
  -name "$IDENTITY_CN" \
  -macalg sha1 \
  -keypbe PBE-SHA1-3DES \
  -certpbe PBE-SHA1-3DES \
  -passout pass:"$P12_PASS" 2>/dev/null

echo "[setup-codesign] importing into $KEYCHAIN"
security import "$WORK/signing.p12" \
  -k "$KEYCHAIN" \
  -P "$P12_PASS" \
  -T /usr/bin/codesign \
  -T /usr/bin/security \
  -A >/dev/null

# Set the partition list on the imported private key. Without this, codesign
# triggers a GUI "allow keychain access" prompt on the first signing attempt
# even though `-T /usr/bin/codesign` was set during import — set-key-partition-list
# is the actual mechanism macOS uses to grant non-interactive access. The
# stderr-warning about "passphrase incorrect" is benign here; the partition
# list still gets updated.
echo "[setup-codesign] enabling non-interactive codesign access"
security set-key-partition-list \
  -S apple-tool:,apple:,codesign: \
  -s -k "" \
  -D "$IDENTITY_CN" \
  -t private \
  "$KEYCHAIN" >/dev/null 2>&1 || true

# Trust the cert for code signing in the user trust domain. Without this,
# `find-identity -p codesigning` reports the identity as untrusted (the
# `(CSSMERR_TP_NOT_TRUSTED)` annotation). codesign itself doesn't actually
# need the trust to function, but adding it keeps find-identity happy and
# avoids any future surprise prompts.
echo "[setup-codesign] adding code-signing trust"
security add-trusted-cert \
  -r trustRoot \
  -p codeSign \
  -k "$KEYCHAIN" \
  "$WORK/signing.crt" >/dev/null 2>&1 || true

# Confirm
if security find-identity -v -p codesigning "$KEYCHAIN" | grep -q "\"$IDENTITY_CN\""; then
  echo "[setup-codesign] done. \"$IDENTITY_CN\" is ready for codesign."
else
  echo "[setup-codesign] WARNING: identity is in the keychain but find-identity reports it as not-yet-trusted." >&2
  echo "[setup-codesign] codesign will still work, but a GUI 'allow' prompt may appear once. After approving" >&2
  echo "[setup-codesign] (or selecting 'Always Allow'), it won't repeat." >&2
fi
