// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Hans-Dieter Zucht. Redistributions must keep this notice and the NOTICE file (author and sources).
// ausgabe — Standard-Ausgabegerät des Systems lesen oder setzen (Core Audio).
// Aufruf: ausgabe            → aktuelles Gerät
//         ausgabe liste      → alle Ausgabegeräte
//         ausgabe "<Name>"   → als Standard-Ausgabe setzen (Teilstring genügt)
import CoreAudio
import Foundation

func prop(_ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}
func devices() -> [AudioDeviceID] {
    var a = prop(kAudioHardwarePropertyDevices); var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &ids)
    return ids
}
func name(_ id: AudioDeviceID) -> String {
    var a = prop(kAudioObjectPropertyName); var n: Unmanaged<CFString>?; var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    AudioObjectGetPropertyData(id, &a, 0, nil, &size, &n)
    return (n?.takeRetainedValue() as String?) ?? "?"
}
func hasOutput(_ id: AudioDeviceID) -> Bool {
    var a = prop(kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput); var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &a, 0, nil, &size)
    return size > 0
}
func current() -> AudioDeviceID {
    var a = prop(kAudioHardwarePropertyDefaultOutputDevice); var id = AudioDeviceID(0); var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, &size, &id)
    return id
}
let args = CommandLine.arguments.dropFirst()
if args.isEmpty { print(name(current())); exit(0) }
if args.first == "liste" { for d in devices() where hasOutput(d) { print(name(d)) }; exit(0) }
let want = args.joined(separator: " ")
guard let d = devices().first(where: { hasOutput($0) && name($0).localizedCaseInsensitiveContains(want) }) else {
    FileHandle.standardError.write("kein Ausgabegerät mit „\(want)“\n".data(using: .utf8)!); exit(1)
}
var a = prop(kAudioHardwarePropertyDefaultOutputDevice); var id = d
let st = AudioObjectSetPropertyData(AudioObjectID(kAudioObjectSystemObject), &a, 0, nil, UInt32(MemoryLayout<AudioDeviceID>.size), &id)
if st != noErr { FileHandle.standardError.write("Setzen scheitert: \(st)\n".data(using: .utf8)!); exit(2) }
print(name(d))
