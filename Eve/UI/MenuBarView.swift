//
//  MenuBarView.swift
//  Eve
//

import CoreAudio
import SwiftUI

struct MenuBarView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            statusSection
            Divider()
            micSection
            Divider()
            controlsSection
            Divider()
            Button("Quit Eve") { NSApplication.shared.terminate(nil) }
                .keyboardShortcut("q")
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
        }
        .frame(width: 280)
    }

    // MARK: - Status

    @ViewBuilder
    private var statusSection: some View {
        HStack {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.headline)
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)

        if appState.status == .noBlackHole {
            Text("BlackHole 2ch not found. Install it and set it as your system mic input in System Settings → Sound.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 12)
                .padding(.bottom, 4)
        } else if appState.status == .processing {
            Text("Set BlackHole 2ch as your mic input in System Settings → Sound and in Zoom, Teams, etc.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 12)
                .padding(.bottom, 4)
        }
    }

    // MARK: - Microphone picker

    @ViewBuilder
    private var micSection: some View {
        Text("Physical Microphone")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
            .padding(.top, 6)
            .padding(.bottom, 2)

        if appState.micPermission.status == .denied {
            Label("Microphone access denied", systemImage: "mic.slash")
                .foregroundStyle(.red)
                .font(.callout)
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
        } else if appState.deviceManager.inputDevices.isEmpty {
            Text("No microphones found")
                .foregroundStyle(.secondary)
                .font(.callout)
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
        } else {
            ForEach(appState.deviceManager.inputDevices) { device in
                Button {
                    appState.selectInputDevice(device.id)
                } label: {
                    HStack {
                        Image(systemName: device.id == appState.deviceManager.selectedInputDeviceID
                              ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(device.id == appState.deviceManager.selectedInputDeviceID
                                             ? .blue : .secondary)
                        Text(device.name).lineLimit(1)
                        Spacer()
                    }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 3)
            }
        }
        Spacer().frame(height: 6)
    }

    // MARK: - Controls

    @ViewBuilder
    private var controlsSection: some View {
        Toggle(isOn: Binding(
            get: { appState.isEnabled },
            set: { appState.toggleEnabled($0) }
        )) {
            Label("Noise Suppression", systemImage: "waveform.badge.minus")
        }
        .toggleStyle(.switch)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .disabled(appState.micPermission.status != .granted)
    }

    // MARK: - Helpers

    private var statusColor: Color {
        switch appState.status {
        case .idle:        return .gray
        case .bypassing:   return .gray
        case .processing:  return .green
        case .error:       return .red
        case .noMicAccess: return .orange
        case .noBlackHole: return .orange
        }
    }

    private var statusText: String {
        switch appState.status {
        case .idle:        return "Idle"
        case .bypassing:   return "Passthrough"
        case .processing:  return "Active"
        case .error:       return "Error"
        case .noMicAccess: return "No Mic Access"
        case .noBlackHole: return "BlackHole Not Found"
        }
    }
}
