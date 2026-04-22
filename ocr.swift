
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
    // have emitted (spaces, units like "m³", stray punctuation).
    let allowed: Set<Character> = Set("0123456789.")
    // Run BOTH levels and keep the longer output. Each level has a failure
    // mode on 7-segment LCDs: .fast can drop narrow glyphs (e.g. a "1"
    // rendered as two segments), while .accurate sometimes returns nothing
    // when confidence is borderline on low-contrast fractional digits.
    // Taking the longer result is a cheap way to catch both cases. The
    // extra ~100 ms is invisible at 10–20 s sampling.
    let fast = recognize(roi, level: .fast).filter { allowed.contains($0) }
    let accurate = recognize(roi, level: .accurate).filter { allowed.contains($0) }
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
func digitsOf(_ s: String) -> String {
    return s.filter { $0.isNumber }
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
