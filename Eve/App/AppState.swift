//
//  AppState.swift
//  Eve
//

import CoreAudio
import Foundation
import Observation

enum EngineStatus {
    case idle
    case bypassing        // engine running, passthrough — toggle is OFF
    case processingIdle   // inference enabled, waiting for an app to open BlackHole
    case processingActive // inference enabled AND an app is actively reading BlackHole
    case error
    case noMicAccess
    case noBlackHole
}

@Observable
final class AppState {
    let deviceManager = AudioDeviceManager()
    let micPermission = MicrophonePermission()

    private(set) var status: EngineStatus = .idle
    private(set) var isEnabled = false

    private let engine = AudioEngineBridge()
    private var statusTimer: Timer?
    private var modelLoaded = false
    private var lastKnownMicID: AudioDeviceID? = nil

    deinit {
        statusTimer?.invalidate()
    }

    init() {
        loadModel()
        requestMicPermissionIfNeeded()
        startStatusPolling()
        // Start passthrough immediately so audio flows as soon as the app launches.
        startEngine()
    }

    func selectInputDevice(_ deviceID: AudioDeviceID) {
        let wasRunning = isEnabled && engine.isRunning
        if wasRunning { engine.stop() }
        deviceManager.selectedInputDeviceID = deviceID
        if wasRunning { startEngine() }
    }

    func toggleEnabled(_ enabled: Bool) {
        isEnabled = enabled
        if engine.isRunning {
            // Engine already running: switch mode without restarting.
            engine.setPassthrough(!enabled)
            if enabled {
                status = engine.hasDemand ? .processingActive : .processingIdle
            } else {
                status = .bypassing
            }
        } else {
            // Engine not yet started: start it (always begins in passthrough until enabled).
            startEngine()
        }
    }

    // MARK: - Private

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "DPDFNet2_16kHz", withExtension: "mlmodelc") else { return }
        modelLoaded = engine.loadModel(at: url)
    }

    private func requestMicPermissionIfNeeded() {
        if micPermission.status == .notDetermined { micPermission.requestAccess() }
    }

    private func startEngine() {
        guard micPermission.status == .granted else { status = .noMicAccess; return }
        guard modelLoaded                       else { status = .error;       return }
        guard let micID = deviceManager.selectedInputDeviceID else { status = .error; return }

        if engine.start(withMicDeviceID: micID) {
            lastKnownMicID = micID
            // Start in passthrough; inference only active when the toggle is on.
            engine.setPassthrough(!isEnabled)
            status = isEnabled ? .processingIdle : .bypassing
        } else {
            status = .noBlackHole
        }
    }

    private func startStatusPolling() {
        statusTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }

            // Restart engine if the selected device changed under us (e.g. USB mic unplugged,
            // causing AudioDeviceManager to fall back to a different device).
            if self.engine.isRunning,
               let currentMic = self.deviceManager.selectedInputDeviceID,
               currentMic != self.lastKnownMicID {
                self.lastKnownMicID = currentMic
                self.startEngine()
                return
            }

            // Auto-start engine once mic permission is granted (handles first-launch permission dialog).
            if !self.engine.isRunning && self.micPermission.status == .granted && self.modelLoaded {
                self.startEngine()
            }
            if self.isEnabled && !self.engine.isRunning { self.status = .error; return }

            // Refresh demand state so the icon updates when an app opens/closes BlackHole.
            if self.isEnabled && self.engine.isRunning {
                self.status = self.engine.hasDemand ? .processingActive : .processingIdle
            }
        }
    }
}
