//
//  StatusIcon.swift
//  Eve
//
//  Menu bar icon that communicates the engine state through color and fill:
//
//    processingActive  — filled waveform, vivid green    (inference running, app is reading)
//    processingIdle    — filled waveform, pale sage green (inference ready, no app reading)
//    bypassing         — outline waveform, amber          (toggle off, passthrough only)
//    error / warning   — outline waveform, red / orange
//    idle              — outline waveform, secondary gray
//

import SwiftUI

struct StatusIcon: View {
    let status: EngineStatus

    var body: some View {
        Image(systemName: symbolName)
            .symbolRenderingMode(.monochrome)
            .foregroundStyle(iconColor)
    }

    // Filled symbol when inference is enabled; outline when off or in an error state.
    private var symbolName: String {
        switch status {
        case .processingActive, .processingIdle:
            return "waveform.circle.fill"
        default:
            return "waveform.circle"
        }
    }

    private var iconColor: Color {
        switch status {
        case .processingActive:
            // Vivid emerald — unambiguous "working" signal.
            return Color(red: 0.18, green: 0.78, blue: 0.44)
        case .processingIdle:
            // Pale sage — same hue family, clearly softer — "ready, waiting".
            return Color(red: 0.56, green: 0.83, blue: 0.65)
        case .bypassing:
            // Warm amber — completely different hue, immediately readable as "standby".
            return Color(red: 0.94, green: 0.74, blue: 0.12)
        case .error:
            return .red
        case .noMicAccess, .noBlackHole:
            return .orange
        case .idle:
            return Color.secondary
        }
    }
}
