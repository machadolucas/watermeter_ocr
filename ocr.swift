
import Foundation
import Vision
import AppKit

// OCR helper for the watermeter project using Apple Vision.
//
// Two modes:
//   --mode digit   (default) — returns a single digit character (backward-compatible
//                              behavior for the mechanical odometer pipeline).
//                              Supports --half full|top|bottom to sub-crop the ROI.
//   --mode line               — returns the full recognized text from the ROI
//                              (digits plus ".") for LCD lines on digital meters.
//                              Falls back from .fast to .accurate on empty output.
//
// Usage: ocr <image> <x> <y> <w> <h> [--half full|top|bottom] [--mode digit|line]
func usage() {
    fputs("usage: ocr <image> <x> <y> <w> <h> [--half full|top|bottom] [--mode digit|line]\n", stderr)
}

let A = CommandLine.arguments
if A.count < 6 { usage(); exit(2) }

let imgPath = A[1]
guard let img = NSImage(contentsOf: URL(fileURLWithPath: imgPath)),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fputs("cannot read image\n", stderr)
    exit(1)
}

let W = CGFloat(cg.width), H = CGFloat(cg.height)
guard let nx = Double(A[2]), let ny = Double(A[3]), let nw = Double(A[4]), let nh = Double(A[5]) else {
    usage(); exit(2)
}

// Parse optional flags in any order.
var half = "full"
var mode = "digit"
var i = 6
while i < A.count {
    let flag = A[i]
    if flag == "--half", i + 1 < A.count {
        half = A[i + 1]; i += 2
    } else if flag == "--mode", i + 1 < A.count {
        mode = A[i + 1]; i += 2
    } else {
        // Unknown or malformed flag — skip so we stay forward-compatible with older callers.
        i += 1
    }
}

var rect = CGRect(x: CGFloat(nx)*W, y: CGFloat(ny)*H, width: CGFloat(nw)*W, height: CGFloat(nh)*H).integral

// --half only applies to single-digit mode. Line mode always uses the full rect.
if mode == "digit" {
    if half == "top" {
        rect.size.height = floor(rect.size.height * 0.55)
    } else if half == "bottom" {
        let h2 = floor(rect.size.height * 0.55)
        rect.origin.y += rect.size.height - h2
        rect.size.height = h2
    }
}

func clamp(_ v: CGFloat, _ lo: CGFloat, _ hi: CGFloat) -> CGFloat { max(lo, min(hi, v)) }
rect.origin.x = clamp(rect.origin.x, 0, W-1)
rect.origin.y = clamp(rect.origin.y, 0, H-1)
rect.size.width = clamp(rect.size.width, 1, W - rect.origin.x)
rect.size.height = clamp(rect.size.height, 1, H - rect.origin.y)

guard let roi = cg.cropping(to: rect) else { print(""); exit(0) }

// Run Vision text recognition at a given level; returns the raw concatenated text
// across all observations (no digit/decimal filtering applied here).
func recognize(_ image: CGImage, level: VNRequestTextRecognitionLevel) -> String {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = level
    req.usesLanguageCorrection = false
    req.recognitionLanguages = ["en-US"]
    req.automaticallyDetectsLanguage = false
    // 0 = disable Vision's built-in small-text filter (default is ~3% of image
    // height in .fast mode). Small 7-segment fractional digits would otherwise
    // be silently dropped from the output.
    req.minimumTextHeight = 0

    // Pin orientation to .up so Vision doesn't silently choose a rotated
    // interpretation when the text is rotationally ambiguous — e.g. a
    // 7-segment "26" whose segments also match "92" when flipped 180°.
    let handler = VNImageRequestHandler(cgImage: image, orientation: .up, options: [:])
    do {
        try handler.perform([req])
    } catch {
        return ""
    }
    let results = req.results ?? []
    var out = ""
    for obs in results {
        if let cand = obs.topCandidates(1).first {
            out += cand.string
        }
    }
    return out
}

if mode == "line" {
    // Keep digits plus the decimal point. Strip everything else Vision might
    // have emitted (spaces, units like "m³", stray punctuation). Also
    // remap common lookalike letters Vision emits for 7-segment glyphs
    // ("O" → "0", etc.) so the filter doesn't throw away a valid read.
    let allowed: Set<Character> = Set("0123456789.")
    // Run BOTH levels and keep the longer output. Each level has a failure
    // mode on 7-segment LCDs: .fast can drop narrow glyphs (e.g. a "1"
    // rendered as two segments), while .accurate sometimes returns nothing
    // when confidence is borderline on low-contrast fractional digits.
    // Taking the longer result is a cheap way to catch both cases. The
    // extra ~100 ms is invisible at 10–20 s sampling.
    func digitize(_ s: String) -> String {
        var out = ""
        for ch in s {
            switch ch {
            case "O", "o", "D", "Q": out.append("0")
            case "I", "l", "|", "i": out.append("1")
            case "Z", "z": out.append("2")
            case "S", "s": out.append("5")
            case "b", "G": out.append("6")
            case "B": out.append("8")
            case "g", "q": out.append("9")
            default:       out.append(ch)
            }
        }
        return out.filter { allowed.contains($0) }
    }
    let fast = digitize(recognize(roi, level: .fast))
    let accurate = digitize(recognize(roi, level: .accurate))
    let text = accurate.count > fast.count ? accurate : fast
    print(text)
    exit(0)
}

// Default: single-digit mode. Used by both the mechanical odometer pipeline
// (large digits, .fast is always enough) and the digital per-digit pipeline
// (smaller digits where .fast sometimes returns nothing on a tight crop).
// Try .fast first for speed; fall back to .accurate when fast couldn't find
// any digit. Matches the "run both and keep the longer one" spirit of line
// mode, but we only need a single character so first non-empty wins.
//
// Vision routinely misclassifies 7-segment "0" as letter "O" / "o" / "D",
// "1" as "I" / "l" / "|", "2" as "Z", "5" as "S", and "6" as "b" /
// "G". These confusions happen because Vision's text recognizer was
// trained on proportional printed text, not segmented LCD glyphs, and
// the hollow center of the 7-segment "0" / "8" closely matches the
// silhouette of "O" / "B". Mapping the common confusions to digits
// BEFORE the digit-only filter turns "OOO" → "000" → first char "0".
func mapLookalikesToDigits(_ s: String) -> String {
    var result = ""
    for ch in s {
        switch ch {
        case "O", "o", "D", "Q": result.append("0")
        case "I", "l", "|", "i": result.append("1")
        case "Z", "z": result.append("2")
        case "S", "s": result.append("5")
        case "b", "G": result.append("6")
        case "B": result.append("8")
        case "g", "q": result.append("9")
        default:       result.append(ch)
        }
    }
    return result
}

func digitsOf(_ s: String) -> String {
    return mapLookalikesToDigits(s).filter { $0.isNumber }
}
var digitsOnly = digitsOf(recognize(roi, level: .fast))
if digitsOnly.isEmpty {
    digitsOnly = digitsOf(recognize(roi, level: .accurate))
}
if digitsOnly.isEmpty {
    print("")
} else {
    print(String(digitsOnly.first!))
}
