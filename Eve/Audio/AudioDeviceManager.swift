//
//  AudioDeviceManager.swift
//  Eve
//

import CoreAudio
import Foundation
import Observation

struct AudioDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let name: String
    let uid: String
}

@Observable
final class AudioDeviceManager {
    private(set) var inputDevices: [AudioDevice] = []
    var selectedInputDeviceID: AudioDeviceID? = nil

    private var listenerBlock: AudioObjectPropertyListenerBlock?

    init() {
        refreshDevices()
        installDeviceChangeListener()
    }

    deinit { removeDeviceChangeListener() }

    func refreshDevices() {
        let all = Self.enumerateAllDevices()
        let isBlackHole = { (d: AudioDevice) in d.name.localizedCaseInsensitiveContains("blackhole") }

        // Physical input devices only — BlackHole is managed internally by the engine.
        inputDevices = all.filter {
            Self.hasStreams($0.id, scope: kAudioObjectPropertyScopeInput) && !isBlackHole($0)
        }

        if selectedInputDeviceID == nil || !inputDevices.contains(where: { $0.id == selectedInputDeviceID }) {
            selectedInputDeviceID =
                Self.defaultDeviceID(selector: kAudioHardwarePropertyDefaultInputDevice)
                    .flatMap { id in inputDevices.first(where: { $0.id == id })?.id }
                ?? inputDevices.first?.id
        }
    }

    // MARK: - Core Audio helpers

    private static func enumerateAllDevices() -> [AudioDevice] {
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject),
                                             &prop, 0, nil, &dataSize) == noErr,
              dataSize > 0 else { return [] }

        let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                         &prop, 0, nil, &dataSize, &ids) == noErr else { return [] }

        return ids.compactMap { id -> AudioDevice? in
            let name = getDeviceName(id) ?? "Unknown"
            let uid  = getDeviceUID(id)  ?? ""
            return AudioDevice(id: id, name: name, uid: uid)
        }
    }

    private static func hasStreams(_ deviceID: AudioDeviceID,
                                   scope: AudioObjectPropertyScope) -> Bool {
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        return AudioObjectGetPropertyDataSize(deviceID, &prop, 0, nil, &dataSize) == noErr
            && dataSize > 0
    }

    private static func getDeviceName(_ deviceID: AudioDeviceID) -> String? {
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: CFString = "" as CFString
        var sz = UInt32(MemoryLayout<CFString>.size)
        return AudioObjectGetPropertyData(deviceID, &prop, 0, nil, &sz, &name) == noErr
            ? name as String : nil
    }

    private static func getDeviceUID(_ deviceID: AudioDeviceID) -> String? {
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var sz = UInt32(MemoryLayout<CFString>.size)
        return AudioObjectGetPropertyData(deviceID, &prop, 0, nil, &sz, &uid) == noErr
            ? uid as String : nil
    }

    private static func defaultDeviceID(selector: AudioObjectPropertySelector) -> AudioDeviceID? {
        var prop = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID: AudioDeviceID = kAudioObjectUnknown
        var sz = UInt32(MemoryLayout<AudioDeviceID>.size)
        let ok = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject),
                                            &prop, 0, nil, &sz, &deviceID) == noErr
        return (ok && deviceID != kAudioObjectUnknown) ? deviceID : nil
    }

    // MARK: - Device change listener

    private func installDeviceChangeListener() {
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            DispatchQueue.main.async { self?.refreshDevices() }
        }
        listenerBlock = block
        AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                            &prop, DispatchQueue.main, block)
    }

    private func removeDeviceChangeListener() {
        guard let block = listenerBlock else { return }
        var prop = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject),
                                               &prop, DispatchQueue.main, block)
    }
}
