
import Foundation
import Vision
import AppKit

// OCR for a single odometer digit using Apple Vision.
// Usage: ocr <image> <x> <y> <w> <h> [--half full|top|bottom]
// Returns a single digit (or empty line).
func usage() {
    fputs("usage: ocr <image> <x> <y> <w> <h> [--half full|top|bottom]\n", stderr)
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

var half = "full"
if A.count >= 8 && A[6] == "--half" { half = A[7] }

var rect = CGRect(x: CGFloat(nx)*W, y: CGFloat(ny)*H, width: CGFloat(nw)*W, height: CGFloat(nh)*H).integral

if half == "top" {
    rect.size.height = floor(rect.size.height * 0.55)
} else if half == "bottom" {
    let h2 = floor(rect.size.height * 0.55)
    rect.origin.y += rect.size.height - h2
    rect.size.height = h2
}

func clamp(_ v: CGFloat, _ lo: CGFloat, _ hi: CGFloat) -> CGFloat { max(lo, min(hi, v)) }
rect.origin.x = clamp(rect.origin.x, 0, W-1)
rect.origin.y = clamp(rect.origin.y, 0, H-1)
rect.size.width = clamp(rect.size.width, 1, W - rect.origin.x)
rect.size.height = clamp(rect.size.height, 1, H - rect.origin.y)

guard let roi = cg.cropping(to: rect) else { print(""); exit(0) }

let req = VNRecognizeTextRequest()
req.recognitionLevel = .fast
req.usesLanguageCorrection = false
req.recognitionLanguages = ["en-US"]
req.automaticallyDetectsLanguage = false

let handler = VNImageRequestHandler(cgImage: roi, options: [:])
do {
    try handler.perform([req])
    let results = (req.results as? [VNRecognizedTextObservation]) ?? []
    var s = ""
    for obs in results {
        if let cand = obs.topCandidates(1).first {
            s += cand.string.filter { $0.isNumber }
        }
    }
    if s.isEmpty {
        print("")
    } else {
        let c = s.first!
        print(String(c))
    }
} catch {
    print("")
}
