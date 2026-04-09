//
//  StatusIcon.swift
//  Eve
//
//  Menu bar icon that indicates whether noise suppression is active.
//

import SwiftUI

struct StatusIcon: View {
    let isActive: Bool

    var body: some View {
        Image(systemName: isActive ? "waveform.circle.fill" : "waveform.circle")
            .symbolRenderingMode(.hierarchical)
    }
}
