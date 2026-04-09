//
//  EveApp.swift
//  Eve
//
//  Real-time noise suppression menu bar app.
//

import SwiftUI

@main
struct EveApp: App {
    @State private var appState = AppState()

    var body: some Scene {
        MenuBarExtra {
            MenuBarView()
                .environment(appState)
        } label: {
            StatusIcon(isActive: appState.status == .processing || appState.status == .bypassing)
        }
        .menuBarExtraStyle(.window)
    }
}
