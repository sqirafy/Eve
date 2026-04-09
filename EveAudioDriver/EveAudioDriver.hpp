#pragma once

#include <CoreAudio/AudioServerPlugIn.h>
#include <atomic>
#include <mach/mach_time.h>

/// Eve Virtual Microphone — CoreAudio HAL AudioServerPlugin.
///
/// Registers a virtual audio input device ("Eve Virtual Mic") that other apps
/// can select as their microphone. Enhanced audio is received from the main
/// Eve app via POSIX shared memory.
///
/// Object hierarchy:
///   kAudioObjectPlugInObject (ID 1) — the plugin
///   └── Device (ID 2) — "Eve Virtual Mic"
///       └── Stream (ID 3) — 1-channel float32 input stream at 48kHz

// Object IDs.
static constexpr AudioObjectID kPlugInObjectID  = kAudioObjectPlugInObject;
static constexpr AudioObjectID kDeviceObjectID  = 2;
static constexpr AudioObjectID kStreamObjectID  = 3;

// Audio format.
static constexpr Float64 kSampleRate       = 48000.0;
static constexpr UInt32  kChannelCount     = 1;
static constexpr UInt32  kBitsPerChannel   = 32;
static constexpr UInt32  kBytesPerFrame    = kChannelCount * (kBitsPerChannel / 8);
static constexpr UInt32  kBufferFrameSize  = 512;

// Device strings.
static constexpr const char* kDeviceName       = "Eve Virtual Mic";
static constexpr const char* kDeviceUID        = "com.eve.virtualmicdevice";
static constexpr const char* kDeviceModelUID   = "com.eve.virtualmicmodel";
static constexpr const char* kManufacturer     = "Eve Audio";

// Forward declarations of the plugin interface functions.
extern "C" {

/// Entry point called by coreaudiod to create the plugin.
void* EveAudioDriverCreate(CFAllocatorRef allocator, CFUUIDRef requestedTypeUUID);

} // extern "C"
