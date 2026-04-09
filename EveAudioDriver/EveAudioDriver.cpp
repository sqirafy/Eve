#include "EveAudioDriver.hpp"
#include "../Eve/Engine/include/SharedAudioBuffer.hpp"

#include <CoreAudio/AudioServerPlugIn.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <cstring>
#include <atomic>
#include <os/log.h>

// ─── Plugin State ──────────────────────────────────────────────────────────

static os_log_t sLog = OS_LOG_DEFAULT;

static std::atomic<UInt32> sRefCount{1};
static AudioServerPlugInHostRef sHost = nullptr;
static eve::SharedAudioBuffer* sSharedBuffer = nullptr;

// IO state
static std::atomic<bool> sIORunning{false};
static UInt64 sIOStartHostTime = 0;
static UInt64 sIOFrameCount = 0;
static mach_timebase_info_data_t sTimebaseInfo;

// ─── Utility ───────────────────────────────────────────────────────────────

static Float64 hostTicksToNanos(UInt64 ticks) {
    return static_cast<Float64>(ticks) *
           static_cast<Float64>(sTimebaseInfo.numer) /
           static_cast<Float64>(sTimebaseInfo.denom);
}

static UInt64 nanosToHostTicks(Float64 nanos) {
    return static_cast<UInt64>(nanos *
           static_cast<Float64>(sTimebaseInfo.denom) /
           static_cast<Float64>(sTimebaseInfo.numer));
}

// ─── Plugin Interface Implementation ───────────────────────────────────────

static HRESULT EveQueryInterface(void* driver, REFIID iid, LPVOID* ppv) {
    CFUUIDRef interfaceID = CFUUIDCreateFromUUIDBytes(nullptr, iid);
    CFUUIDRef pluginIID = CFUUIDGetConstantUUIDWithBytes(nullptr,
        0x44, 0x3A, 0xBC, 0xAD, 0xEB, 0x73, 0x11, 0xD5,
        0x94, 0x60, 0x00, 0x30, 0x65, 0x6D, 0x85, 0x2C);

    if (CFEqual(interfaceID, pluginIID) ||
        CFEqual(interfaceID, IUnknownUUID)) {
        sRefCount.fetch_add(1, std::memory_order_relaxed);
        *ppv = driver;
        CFRelease(interfaceID);
        return S_OK;
    }

    *ppv = nullptr;
    CFRelease(interfaceID);
    return E_NOINTERFACE;
}

static ULONG EveAddRef(void* /*driver*/) {
    return sRefCount.fetch_add(1, std::memory_order_relaxed) + 1;
}

static ULONG EveRelease(void* /*driver*/) {
    UInt32 count = sRefCount.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (count == 0) {
        if (sSharedBuffer) {
            eve::closeSharedAudioBuffer(sSharedBuffer);
            sSharedBuffer = nullptr;
        }
    }
    return count;
}

// ─── Initialize / Teardown ────────────────────────────────────────────────

static OSStatus EveInitialize(AudioServerPlugInDriverRef /*driver*/,
                               AudioServerPlugInHostRef host) {
    sHost = host;
    mach_timebase_info(&sTimebaseInfo);

    // Open shared memory segment (created by the main Eve app).
    sSharedBuffer = eve::openSharedAudioBuffer(false);
    if (!sSharedBuffer) {
        os_log_error(sLog, "Eve driver: failed to open shared memory");
        // Not fatal — we'll output silence until the app creates it.
    }

    return kAudioHardwareNoError;
}

static OSStatus EveCreateDevice(AudioServerPlugInDriverRef, CFDictionaryRef,
                                 const AudioServerPlugInClientInfo*,
                                 AudioObjectID*) {
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus EveDestroyDevice(AudioServerPlugInDriverRef, AudioObjectID) {
    return kAudioHardwareUnsupportedOperationError;
}

// ─── Property Helpers ─────────────────────────────────────────────────────

static CFStringRef CreateCFString(const char* str) {
    return CFStringCreateWithCString(nullptr, str, kCFStringEncodingUTF8);
}

// ─── HasProperty ──────────────────────────────────────────────────────────

static Boolean EveHasProperty(AudioServerPlugInDriverRef,
                               AudioObjectID objectID,
                               pid_t,
                               const AudioObjectPropertyAddress* address) {
    switch (objectID) {
    case kPlugInObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyManufacturer:
        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
        case kAudioPlugInPropertyTranslateUIDToDevice:
        case kAudioPlugInPropertyResourceBundle:
            return true;
        default:
            return false;
        }

    case kDeviceObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioDevicePropertyDeviceUID:
        case kAudioDevicePropertyModelUID:
        case kAudioDevicePropertyTransportType:
        case kAudioDevicePropertyRelatedDevices:
        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceIsRunning:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertyStreams:
        case kAudioObjectPropertyControlList:
        case kAudioDevicePropertyNominalSampleRate:
        case kAudioDevicePropertyAvailableNominalSampleRates:
        case kAudioDevicePropertyZeroTimeStampPeriod:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyPreferredChannelsForStereo:
        case kAudioDevicePropertyPreferredChannelLayout:
            return true;
        default:
            return false;
        }

    case kStreamObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
        case kAudioStreamPropertyIsActive:
        case kAudioStreamPropertyDirection:
        case kAudioStreamPropertyTerminalType:
        case kAudioStreamPropertyStartingChannel:
        case kAudioStreamPropertyLatency:
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            return true;
        default:
            return false;
        }

    default:
        return false;
    }
}

// ─── IsPropertySettable ───────────────────────────────────────────────────

static OSStatus EveIsPropertySettable(AudioServerPlugInDriverRef,
                                       AudioObjectID, pid_t,
                                       const AudioObjectPropertyAddress*,
                                       Boolean* outIsSettable) {
    // None of our properties are settable.
    *outIsSettable = false;
    return kAudioHardwareNoError;
}

// ─── GetPropertyDataSize ──────────────────────────────────────────────────

static OSStatus EveGetPropertyDataSize(AudioServerPlugInDriverRef,
                                        AudioObjectID objectID,
                                        pid_t,
                                        const AudioObjectPropertyAddress* address,
                                        UInt32,
                                        const void*,
                                        UInt32* outDataSize) {
    switch (objectID) {
    case kPlugInObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioClassID); return kAudioHardwareNoError;
        case kAudioObjectPropertyManufacturer:
        case kAudioPlugInPropertyResourceBundle:
            *outDataSize = sizeof(CFStringRef); return kAudioHardwareNoError;
        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
            *outDataSize = sizeof(AudioObjectID); return kAudioHardwareNoError;
        case kAudioPlugInPropertyTranslateUIDToDevice:
            *outDataSize = sizeof(AudioObjectID); return kAudioHardwareNoError;
        default:
            *outDataSize = 0; return kAudioHardwareUnknownPropertyError;
        }

    case kDeviceObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID); return kAudioHardwareNoError;
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID); return kAudioHardwareNoError;
        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioDevicePropertyDeviceUID:
        case kAudioDevicePropertyModelUID:
            *outDataSize = sizeof(CFStringRef); return kAudioHardwareNoError;
        case kAudioDevicePropertyTransportType:
        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyZeroTimeStampPeriod:
            *outDataSize = sizeof(UInt32); return kAudioHardwareNoError;
        case kAudioDevicePropertyRelatedDevices:
        case kAudioDevicePropertyStreams:
            *outDataSize = sizeof(AudioObjectID); return kAudioHardwareNoError;
        case kAudioObjectPropertyControlList:
            *outDataSize = 0; return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceIsRunning:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
            *outDataSize = sizeof(UInt32); return kAudioHardwareNoError;
        case kAudioDevicePropertyNominalSampleRate:
            *outDataSize = sizeof(Float64); return kAudioHardwareNoError;
        case kAudioDevicePropertyAvailableNominalSampleRates:
            *outDataSize = sizeof(AudioValueRange); return kAudioHardwareNoError;
        case kAudioDevicePropertyPreferredChannelsForStereo:
            *outDataSize = 2 * sizeof(UInt32); return kAudioHardwareNoError;
        case kAudioDevicePropertyPreferredChannelLayout:
            *outDataSize = offsetof(AudioChannelLayout, mChannelDescriptions) +
                           kChannelCount * sizeof(AudioChannelDescription);
            return kAudioHardwareNoError;
        default:
            *outDataSize = 0; return kAudioHardwareUnknownPropertyError;
        }

    case kStreamObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID); return kAudioHardwareNoError;
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID); return kAudioHardwareNoError;
        case kAudioStreamPropertyIsActive:
        case kAudioStreamPropertyDirection:
        case kAudioStreamPropertyTerminalType:
        case kAudioStreamPropertyStartingChannel:
        case kAudioStreamPropertyLatency:
            *outDataSize = sizeof(UInt32); return kAudioHardwareNoError;
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
            *outDataSize = sizeof(AudioStreamBasicDescription); return kAudioHardwareNoError;
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            *outDataSize = sizeof(AudioStreamRangedDescription); return kAudioHardwareNoError;
        default:
            *outDataSize = 0; return kAudioHardwareUnknownPropertyError;
        }

    default:
        *outDataSize = 0;
        return kAudioHardwareBadObjectError;
    }
}

// ─── GetPropertyData ──────────────────────────────────────────────────────

static AudioStreamBasicDescription makeStreamFormat() {
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = kSampleRate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kAudioFormatFlagIsFloat |
                       kAudioFormatFlagsNativeEndian |
                       kAudioFormatFlagIsPacked;
    fmt.mBitsPerChannel = kBitsPerChannel;
    fmt.mChannelsPerFrame = kChannelCount;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = kBytesPerFrame;
    fmt.mBytesPerPacket = kBytesPerFrame;
    return fmt;
}

static OSStatus EveGetPropertyData(AudioServerPlugInDriverRef,
                                    AudioObjectID objectID,
                                    pid_t,
                                    const AudioObjectPropertyAddress* address,
                                    UInt32 qualifierDataSize,
                                    const void* qualifierData,
                                    UInt32 inDataSize,
                                    UInt32* outDataSize,
                                    void* outData) {
    switch (objectID) {
    case kPlugInObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioObjectClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioPlugInClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = kAudioObjectUnknown;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyManufacturer:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CreateCFString(kManufacturer);
            return kAudioHardwareNoError;
        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = kDeviceObjectID;
            return kAudioHardwareNoError;
        case kAudioPlugInPropertyTranslateUIDToDevice: {
            CFStringRef uid = *static_cast<const CFStringRef*>(qualifierData);
            CFStringRef deviceUID = CreateCFString(kDeviceUID);
            AudioObjectID result = kAudioObjectUnknown;
            if (CFStringCompare(uid, deviceUID, 0) == kCFCompareEqualTo) {
                result = kDeviceObjectID;
            }
            CFRelease(deviceUID);
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = result;
            return kAudioHardwareNoError;
        }
        case kAudioPlugInPropertyResourceBundle:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CFSTR("");
            return kAudioHardwareNoError;
        default:
            return kAudioHardwareUnknownPropertyError;
        }

    case kDeviceObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioObjectClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioDeviceClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = kPlugInObjectID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyName:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CreateCFString(kDeviceName);
            return kAudioHardwareNoError;
        case kAudioObjectPropertyManufacturer:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CreateCFString(kManufacturer);
            return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceUID:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CreateCFString(kDeviceUID);
            return kAudioHardwareNoError;
        case kAudioDevicePropertyModelUID:
            *outDataSize = sizeof(CFStringRef);
            *static_cast<CFStringRef*>(outData) = CreateCFString(kDeviceModelUID);
            return kAudioHardwareNoError;
        case kAudioDevicePropertyTransportType:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = kAudioDeviceTransportTypeVirtual;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyRelatedDevices:
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = kDeviceObjectID;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyClockDomain:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceIsAlive:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 1;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceIsRunning:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = sIORunning.load() ? 1 : 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
            *outDataSize = sizeof(UInt32);
            // Allow as default input device.
            *static_cast<UInt32*>(outData) =
                (address->mScope == kAudioObjectPropertyScopeInput) ? 1 : 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyLatency:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyStreams:
            if (address->mScope == kAudioObjectPropertyScopeInput ||
                address->mScope == kAudioObjectPropertyScopeGlobal) {
                *outDataSize = sizeof(AudioObjectID);
                *static_cast<AudioObjectID*>(outData) = kStreamObjectID;
            } else {
                *outDataSize = 0;
            }
            return kAudioHardwareNoError;
        case kAudioObjectPropertyControlList:
            *outDataSize = 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyNominalSampleRate:
            *outDataSize = sizeof(Float64);
            *static_cast<Float64*>(outData) = kSampleRate;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyAvailableNominalSampleRates: {
            AudioValueRange range = { kSampleRate, kSampleRate };
            *outDataSize = sizeof(range);
            std::memcpy(outData, &range, sizeof(range));
            return kAudioHardwareNoError;
        }
        case kAudioDevicePropertySafetyOffset:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 0;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyZeroTimeStampPeriod:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = kBufferFrameSize;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyPreferredChannelsForStereo:
            *outDataSize = 2 * sizeof(UInt32);
            static_cast<UInt32*>(outData)[0] = 1;
            static_cast<UInt32*>(outData)[1] = 1;
            return kAudioHardwareNoError;
        case kAudioDevicePropertyPreferredChannelLayout: {
            auto* layout = static_cast<AudioChannelLayout*>(outData);
            layout->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
            layout->mChannelBitmap = 0;
            layout->mNumberChannelDescriptions = kChannelCount;
            layout->mChannelDescriptions[0].mChannelLabel = kAudioChannelLabel_Mono;
            layout->mChannelDescriptions[0].mChannelFlags = 0;
            layout->mChannelDescriptions[0].mCoordinates[0] = 0;
            layout->mChannelDescriptions[0].mCoordinates[1] = 0;
            layout->mChannelDescriptions[0].mCoordinates[2] = 0;
            *outDataSize = offsetof(AudioChannelLayout, mChannelDescriptions) +
                           kChannelCount * sizeof(AudioChannelDescription);
            return kAudioHardwareNoError;
        }
        default:
            return kAudioHardwareUnknownPropertyError;
        }

    case kStreamObjectID:
        switch (address->mSelector) {
        case kAudioObjectPropertyBaseClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioObjectClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            *static_cast<AudioClassID*>(outData) = kAudioStreamClassID;
            return kAudioHardwareNoError;
        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            *static_cast<AudioObjectID*>(outData) = kDeviceObjectID;
            return kAudioHardwareNoError;
        case kAudioStreamPropertyIsActive:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 1;
            return kAudioHardwareNoError;
        case kAudioStreamPropertyDirection:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 1; // 1 = input (from the perspective of clients)
            return kAudioHardwareNoError;
        case kAudioStreamPropertyTerminalType:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = kAudioStreamTerminalTypeMicrophone;
            return kAudioHardwareNoError;
        case kAudioStreamPropertyStartingChannel:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 1;
            return kAudioHardwareNoError;
        case kAudioStreamPropertyLatency:
            *outDataSize = sizeof(UInt32);
            *static_cast<UInt32*>(outData) = 0;
            return kAudioHardwareNoError;
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat: {
            auto fmt = makeStreamFormat();
            *outDataSize = sizeof(fmt);
            std::memcpy(outData, &fmt, sizeof(fmt));
            return kAudioHardwareNoError;
        }
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats: {
            AudioStreamRangedDescription ranged = {};
            ranged.mFormat = makeStreamFormat();
            ranged.mSampleRateRange = { kSampleRate, kSampleRate };
            *outDataSize = sizeof(ranged);
            std::memcpy(outData, &ranged, sizeof(ranged));
            return kAudioHardwareNoError;
        }
        default:
            return kAudioHardwareUnknownPropertyError;
        }

    default:
        return kAudioHardwareBadObjectError;
    }
}

// ─── SetPropertyData ──────────────────────────────────────────────────────

static OSStatus EveSetPropertyData(AudioServerPlugInDriverRef,
                                    AudioObjectID, pid_t,
                                    const AudioObjectPropertyAddress*,
                                    UInt32, const void*, UInt32, const void*) {
    return kAudioHardwareUnsupportedOperationError;
}

// ─── IO Operations ────────────────────────────────────────────────────────

static OSStatus EveStartIO(AudioServerPlugInDriverRef,
                            AudioObjectID deviceID,
                            UInt32 /*clientID*/) {
    if (deviceID != kDeviceObjectID) return kAudioHardwareBadObjectError;

    sIOStartHostTime = mach_absolute_time();
    sIOFrameCount = 0;
    sIORunning.store(true, std::memory_order_release);

    // Try to open shared memory if not already open.
    if (!sSharedBuffer) {
        sSharedBuffer = eve::openSharedAudioBuffer(false);
    }

    // Notify the main app that a client is reading.
    if (sSharedBuffer) {
        sSharedBuffer->active_clients.fetch_add(1, std::memory_order_release);
    }

    return kAudioHardwareNoError;
}

static OSStatus EveStopIO(AudioServerPlugInDriverRef,
                           AudioObjectID deviceID,
                           UInt32 /*clientID*/) {
    if (deviceID != kDeviceObjectID) return kAudioHardwareBadObjectError;

    sIORunning.store(false, std::memory_order_release);

    if (sSharedBuffer) {
        sSharedBuffer->active_clients.fetch_sub(1, std::memory_order_release);
    }

    return kAudioHardwareNoError;
}

static OSStatus EveGetZeroTimeStamp(AudioServerPlugInDriverRef,
                                     AudioObjectID deviceID,
                                     UInt32 /*clientID*/,
                                     Float64* outSampleTime,
                                     UInt64* outHostTime,
                                     UInt64* outSeed) {
    if (deviceID != kDeviceObjectID) return kAudioHardwareBadObjectError;

    // Compute timestamps based on sample rate and buffer size.
    UInt64 currentHostTime = mach_absolute_time();
    Float64 elapsedNanos = hostTicksToNanos(currentHostTime - sIOStartHostTime);
    Float64 elapsedSamples = (elapsedNanos / 1e9) * kSampleRate;

    // Align to buffer boundary.
    UInt64 bufferCycles = static_cast<UInt64>(elapsedSamples) / kBufferFrameSize;
    Float64 sampleTime = static_cast<Float64>(bufferCycles * kBufferFrameSize);

    Float64 sampleNanos = (sampleTime / kSampleRate) * 1e9;
    UInt64 hostTime = sIOStartHostTime + nanosToHostTicks(sampleNanos);

    *outSampleTime = sampleTime;
    *outHostTime = hostTime;
    *outSeed = 1;

    return kAudioHardwareNoError;
}

static OSStatus EveWillDoIOOperation(AudioServerPlugInDriverRef,
                                      AudioObjectID, UInt32,
                                      UInt32 operationID,
                                      Boolean* outWillDo,
                                      Boolean* outWillDoInPlace) {
    *outWillDo = (operationID == kAudioServerPlugInIOOperationReadInput);
    *outWillDoInPlace = true;
    return kAudioHardwareNoError;
}

static OSStatus EveBeginIOOperation(AudioServerPlugInDriverRef,
                                     AudioObjectID, UInt32, UInt32,
                                     UInt32, const AudioServerPlugInIOCycleInfo*) {
    return kAudioHardwareNoError;
}

static OSStatus EveDoIOOperation(AudioServerPlugInDriverRef,
                                  AudioObjectID deviceID,
                                  AudioObjectID streamID,
                                  UInt32 /*clientID*/,
                                  UInt32 operationID,
                                  UInt32 ioBufferFrameSize,
                                  const AudioServerPlugInIOCycleInfo* /*ioCycleInfo*/,
                                  void* ioMainBuffer,
                                  void* /*ioSecondaryBuffer*/) {
    if (operationID != kAudioServerPlugInIOOperationReadInput) {
        return kAudioHardwareNoError;
    }

    auto* outBuffer = static_cast<float*>(ioMainBuffer);

    if (sSharedBuffer) {
        // Pop enhanced audio from shared memory.
        uint64_t got = sSharedBuffer->pop(outBuffer, ioBufferFrameSize);

        // If we didn't get enough samples, zero-fill the remainder.
        if (got < ioBufferFrameSize) {
            std::memset(outBuffer + got, 0,
                        (ioBufferFrameSize - got) * sizeof(float));
        }
    } else {
        // No shared memory — output silence.
        std::memset(outBuffer, 0, ioBufferFrameSize * sizeof(float));
    }

    return kAudioHardwareNoError;
}

static OSStatus EveEndIOOperation(AudioServerPlugInDriverRef,
                                   AudioObjectID, UInt32, UInt32,
                                   UInt32, const AudioServerPlugInIOCycleInfo*) {
    return kAudioHardwareNoError;
}

// ─── Interface VTable ─────────────────────────────────────────────────────

static AudioServerPlugInDriverInterface sDriverInterface = {
    // Reserved
    nullptr,
    // IUnknown
    EveQueryInterface,
    EveAddRef,
    EveRelease,
    // Plugin lifecycle
    EveInitialize,
    EveCreateDevice,
    EveDestroyDevice,
    nullptr, // AddDeviceClient
    nullptr, // RemoveDeviceClient
    nullptr, // PerformDeviceConfigurationChange
    nullptr, // AbortDeviceConfigurationChange
    // Properties
    EveHasProperty,
    EveIsPropertySettable,
    EveGetPropertyDataSize,
    EveGetPropertyData,
    EveSetPropertyData,
    // IO
    EveStartIO,
    EveStopIO,
    EveGetZeroTimeStamp,
    EveWillDoIOOperation,
    EveBeginIOOperation,
    EveDoIOOperation,
    EveEndIOOperation,
};

static AudioServerPlugInDriverInterface* sDriverInterfacePtr = &sDriverInterface;

// ─── Entry Point ──────────────────────────────────────────────────────────

void* EveAudioDriverCreate(CFAllocatorRef /*allocator*/, CFUUIDRef requestedTypeUUID) {
    CFUUIDRef typeID = CFUUIDGetConstantUUIDWithBytes(nullptr,
        0x44, 0x3A, 0xBC, 0xAB, 0xEB, 0x73, 0x11, 0xD5,
        0x94, 0x60, 0x00, 0x30, 0x65, 0x6D, 0x85, 0x2C);

    if (!CFEqual(requestedTypeUUID, typeID)) {
        return nullptr;
    }

    sLog = os_log_create("com.eve.audiodriver", "driver");

    return &sDriverInterfacePtr;
}
