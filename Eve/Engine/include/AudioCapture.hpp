#pragma once

#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <atomic>
#include <cstdint>

namespace eve {

template <typename T>
class SPSCRingBuffer;

/// Captures audio from a hardware input device using Core Audio HAL.
/// The IO proc callback is real-time safe: no allocations, no locks, no ObjC.
class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    AudioCapture(const AudioCapture&) = delete;
    AudioCapture& operator=(const AudioCapture&) = delete;

    /// Set the ring buffer that captured samples will be pushed to.
    void setOutputBuffer(SPSCRingBuffer<float>* buffer);

    /// Set a callback invoked (from the real-time IO thread) after samples are
    /// pushed to the ring buffer.  Used to signal the inference worker.
    using NotifyCallback = void (*)(void* ctx);
    void setNotifyCallback(NotifyCallback cb, void* ctx);

    /// Start capturing from the given device at the target sample rate.
    /// Returns true on success.
    bool start(AudioDeviceID deviceID, Float64 targetSampleRate = 48000.0);

    /// Stop capturing.
    void stop();

    /// Whether capture is currently active.
    bool isRunning() const;

    /// Get the actual sample rate of the device after start.
    Float64 actualSampleRate() const { return actual_sample_rate_; }

private:
    /// The real-time IO proc callback registered with Core Audio.
    static OSStatus ioProc(AudioDeviceID device,
                           const AudioTimeStamp* now,
                           const AudioBufferList* inputData,
                           const AudioTimeStamp* inputTime,
                           AudioBufferList* outputData,
                           const AudioTimeStamp* outputTime,
                           void* clientData);

    /// Configure the device buffer size for low latency.
    bool configureDevice(AudioDeviceID deviceID, Float64 targetSampleRate);

    AudioDeviceID device_id_ = kAudioObjectUnknown;
    AudioDeviceIOProcID io_proc_id_ = nullptr;
    SPSCRingBuffer<float>* output_buffer_ = nullptr;

    NotifyCallback notify_cb_  = nullptr;
    void*          notify_ctx_ = nullptr;

    // Sample rate conversion (when device rate != target rate)
    AudioConverterRef converter_ = nullptr;
    Float64 target_sample_rate_ = 48000.0;
    Float64 actual_sample_rate_ = 48000.0;
    UInt32 device_channels_ = 1;

    // Intermediate buffer for sample rate conversion
    static constexpr size_t kMaxConvertFrames = 4096;
    float convert_buffer_[kMaxConvertFrames] = {};

    std::atomic<bool> running_{false};
};

} // namespace eve
