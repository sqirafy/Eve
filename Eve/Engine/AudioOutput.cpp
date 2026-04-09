#include "AudioOutput.hpp"
#include "SPSCRingBuffer.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>

namespace eve {

// ─── Device discovery ────────────────────────────────────────────────────────

AudioDeviceID findOutputDeviceByName(const char* keyword) {
    AudioObjectPropertyAddress prop = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    UInt32 dataSize = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &prop,
                                       0, nullptr, &dataSize) != noErr) {
        return kAudioObjectUnknown;
    }

    const int count = static_cast<int>(dataSize / sizeof(AudioDeviceID));
    AudioDeviceID ids[256] = {};
    if (count > 256) return kAudioObjectUnknown;

    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &prop,
                                   0, nullptr, &dataSize, ids) != noErr) {
        return kAudioObjectUnknown;
    }

    AudioObjectPropertyAddress nameProp = {
        kAudioObjectPropertyName,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    for (int i = 0; i < count; ++i) {
        CFStringRef cfName = nullptr;
        UInt32 sz = sizeof(cfName);
        if (AudioObjectGetPropertyData(ids[i], &nameProp, 0, nullptr, &sz, &cfName) != noErr) {
            continue;
        }
        char buf[256] = {};
        bool matched = false;
        if (CFStringGetCString(cfName, buf, sizeof(buf), kCFStringEncodingUTF8)) {
            // Case-insensitive substring search
            std::string name(buf);
            std::string kw(keyword);
            auto it = std::search(name.begin(), name.end(),
                                  kw.begin(), kw.end(),
                                  [](char a, char b){ return std::tolower(a) == std::tolower(b); });
            matched = (it != name.end());
        }
        CFRelease(cfName);
        if (matched) return ids[i];
    }
    return kAudioObjectUnknown;
}

// ─── AudioOutput ─────────────────────────────────────────────────────────────

AudioOutput::AudioOutput() = default;

AudioOutput::~AudioOutput() {
    stop();
}

void AudioOutput::setInputBuffer(SPSCRingBuffer<float>* buffer) {
    input_buffer_ = buffer;
}

bool AudioOutput::start(AudioDeviceID deviceID, Float64 sampleRate) {
    if (running_.load(std::memory_order_relaxed)) stop();
    if (!input_buffer_) return false;

    device_id_ = deviceID;

    // Try to set the device sample rate to match the engine (48 kHz).
    AudioObjectPropertyAddress rateProp = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectSetPropertyData(deviceID, &rateProp, 0, nullptr,
                               sizeof(sampleRate), &sampleRate);

    OSStatus err = AudioDeviceCreateIOProcID(device_id_, ioProc, this, &io_proc_id_);
    if (err != noErr) return false;

    err = AudioDeviceStart(device_id_, io_proc_id_);
    if (err != noErr) {
        AudioDeviceDestroyIOProcID(device_id_, io_proc_id_);
        io_proc_id_ = nullptr;
        return false;
    }

    running_.store(true, std::memory_order_release);
    return true;
}

void AudioOutput::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);

    if (device_id_ != kAudioObjectUnknown && io_proc_id_) {
        AudioDeviceStop(device_id_, io_proc_id_);
        AudioDeviceDestroyIOProcID(device_id_, io_proc_id_);
    }
    io_proc_id_ = nullptr;
    device_id_  = kAudioObjectUnknown;
}

bool AudioOutput::isRunning() const {
    return running_.load(std::memory_order_acquire);
}

// Real-time output callback — NO allocations, NO locks, NO ObjC.
OSStatus AudioOutput::ioProc(AudioDeviceID /*device*/,
                              const AudioTimeStamp* /*now*/,
                              const AudioBufferList* /*inputData*/,
                              const AudioTimeStamp* /*inputTime*/,
                              AudioBufferList*       outputData,
                              const AudioTimeStamp* /*outputTime*/,
                              void*                  clientData)
{
    auto* self = static_cast<AudioOutput*>(clientData);

    if (!outputData || outputData->mNumberBuffers == 0) return noErr;

    AudioBuffer& buf = outputData->mBuffers[0];
    auto*        dst = static_cast<float*>(buf.mData);
    const UInt32 channels   = buf.mNumberChannels;
    const UInt32 frameCount = buf.mDataByteSize / (channels * sizeof(float));

    if (!self->input_buffer_ || !dst || frameCount == 0) {
        if (dst) std::memset(dst, 0, buf.mDataByteSize);
        return noErr;
    }

    // Pop mono samples from the processed ring buffer.
    float mono[2048];
    const UInt32 safe = std::min(frameCount, UInt32(2048));
    const size_t got  = self->input_buffer_->pop(mono, safe);

    // Zero-fill any gap (inference thread not yet warm).
    if (got < safe) std::memset(mono + got, 0, (safe - got) * sizeof(float));

    // Broadcast mono → all output channels (BlackHole-2ch is stereo).
    for (UInt32 f = 0; f < safe; ++f) {
        for (UInt32 c = 0; c < channels; ++c) {
            dst[f * channels + c] = mono[f];
        }
    }

    return noErr;
}

} // namespace eve
