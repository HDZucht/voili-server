// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hans-Dieter Zucht. Redistributions must keep this notice and the NOTICE file (author and sources).
// klang — erkennt Musik und Sprache im Live-Ton mit Apples SoundAnalysis (eingebauter Klassifikator).
// Ends:  HDZ 2026-11-22: „könnte das Modell erkennen, wenn Musik da ist, und den Mix bei Musik umstellen?“ ·
//        „Musik-Switcher, zwei Regler Sprachannahme (5 %) und Musikannahme 15 % als Default, beide wählbar“.
// Ways:  liest Float32 mono 48 kHz von stdin, schreibt je Analysefenster eine JSON-Zeile auf stdout:
//        {"musik":0.82,"sprache":0.10,"gesang":0.03}. `klang klassen` listet die Bezeichner des Klassifikators.
// Means: SNAudioStreamAnalyzer + SNClassifySoundRequest(.version1), läuft ohne Mikrofonrecht (nur Puffer).
// Fails: Fehler des Analysators auf stderr; Ende von stdin beendet das Programm.
import AVFoundation
import Foundation
import SoundAnalysis

let fmt = AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1)!
let req = try SNClassifySoundRequest(classifierIdentifier: .version1)
if CommandLine.arguments.dropFirst().first == "klassen" {
    print(req.knownClassifications.sorted().joined(separator: "\n")); exit(0)
}
req.overlapFactor = 0.5

final class Beobachter: NSObject, SNResultsObserving {
    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let r = result as? SNClassificationResult else { return }
        func c(_ id: String) -> Double { r.classification(forIdentifier: id)?.confidence ?? 0 }
        print(String(format: "{\"musik\":%.3f,\"sprache\":%.3f,\"gesang\":%.3f}", c("music"), c("speech"), c("singing")))
        fflush(stdout)
    }
    func request(_ request: SNRequest, didFailWithError error: Error) {
        FileHandle.standardError.write("klang: \(error)\n".data(using: .utf8)!)
    }
}

let analysator = SNAudioStreamAnalyzer(format: fmt)
let beobachter = Beobachter()
try analysator.add(req, withObserver: beobachter)
var pos: AVAudioFramePosition = 0
let eingabe = FileHandle.standardInput
while true {
    let data = eingabe.readData(ofLength: 4800 * 4)
    if data.isEmpty { break }
    let n = data.count / 4
    if n == 0 { continue }
    let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n))!
    buf.frameLength = AVAudioFrameCount(n)
    data.withUnsafeBytes { raw in memcpy(buf.floatChannelData![0], raw.baseAddress!, n * 4) }
    analysator.analyze(buf, atAudioFramePosition: pos)
    pos += AVAudioFramePosition(n)
}
analysator.completeAnalysis()
