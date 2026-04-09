//
//  MicrophonePermission.swift
//  Eve
//
//  Manages microphone access permission via AVCaptureDevice.
//

import AVFoundation
import Observation

@Observable
final class MicrophonePermission {
    enum Status {
        case notDetermined
        case granted
        case denied
    }

    private(set) var status: Status = .notDetermined

    init() {
        updateStatus()
    }

    func requestAccess() {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            DispatchQueue.main.async {
                self?.status = granted ? .granted : .denied
            }
        }
    }

    private func updateStatus() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            status = .granted
        case .denied, .restricted:
            status = .denied
        case .notDetermined:
            status = .notDetermined
        @unknown default:
            status = .notDetermined
        }
    }
}
