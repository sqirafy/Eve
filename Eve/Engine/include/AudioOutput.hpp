#pragma once

#include <CoreAudio/CoreAudio.h>
#include <atomic>

namespace eve {

template <typename T>
class SPSCRingBuffer;

/// Writes mono float32 audio from a ring buffer to a CoreAudio output device
/// (e.g. BlackHole) in real-time via an AudioDeviceIOProc.
class AudioOutput {
public:
    AudioOutput();
    ~AudioOutput();

    AudioOutput(const AudioOutput&) = delete;
    AudioOutput& operator=(const AudioOutput&) = delete;

    void setInputBuffer(SPSCRingBuffer<float>* buffer);

    /// Start writing to the given output device at the given sample rate.
    bool start(AudioDeviceID deviceID, Float64 sampleRate);

    void stop();
    bool isRunning() const;

private:
    static OSStatus ioProc(AudioDeviceID,
                           const AudioTimeStamp*,
                           const AudioBufferList*,
                           const AudioTimeStamp*,
                           AudioBufferList*       outputData,
                           const AudioTimeStamp*,
                           void*                  clientData);

    SPSCRingBuffer<float>* input_buffer_ = nullptr;
    AudioDeviceID          device_id_    = kAudioObjectUnknown;
    AudioDeviceIOProcID    io_proc_id_   = nullptr;
    std::atomic<bool>      running_{false};
};

/// Return the AudioDeviceID of the first device whose name contains `keyword`,
/// or kAudioObjectUnknown if not found.
AudioDeviceID findOutputDeviceByName(const char* keyword);

} // namespace eve
