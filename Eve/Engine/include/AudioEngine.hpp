#pragma once

#include <CoreAudio/CoreAudio.h>
#include <dispatch/dispatch.h>
#include <memory>
#include <atomic>

namespace eve {

template <typename T>
class SPSCRingBuffer;

class AudioCapture;
class AudioOutput;
class InferenceWorker;

/// Audio pipeline:
///   Physical mic → capture_buffer_ → InferenceWorker → processed_buffer_ → BlackHole
///
/// BlackHole acts as a virtual microphone: recording apps that select BlackHole
/// as their input receive the denoised audio Eve writes to it.
///
/// Speaker output is NOT routed through Eve — set your system output directly
/// to your physical speakers/headphones in System Settings.
class AudioEngine {
public:
    AudioEngine();
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    bool loadModel(const char* modelPath);

    /// Start the pipeline. micDeviceID is the physical microphone to capture from.
    /// BlackHole is auto-detected as the output destination.
    bool start(AudioDeviceID micDeviceID);

    void stop();
    bool isRunning() const;
    bool isBlackHoleConnected() const;

    /// Switch between passthrough (no inference) and active denoising on the fly.
    void setPassthrough(bool passthrough);

private:
    static constexpr size_t kBufCapacity = 32768;

    std::unique_ptr<SPSCRingBuffer<float>> capture_buffer_;
    std::unique_ptr<SPSCRingBuffer<float>> processed_buffer_;
    std::unique_ptr<AudioCapture>          mic_capture_;
    std::unique_ptr<InferenceWorker>       worker_;
    std::unique_ptr<AudioOutput>           blackhole_output_;

    void* model_context_ = nullptr;
    std::atomic<bool> running_{false};

    // Demand-based inference: only run CoreML when an app reads from BlackHole.
    AudioDeviceID blackhole_id_ = kAudioObjectUnknown;
    std::atomic<bool> has_demand_{false};

    void startDemandMonitoring();
    void stopDemandMonitoring();
    void onDemandStarted();

    static OSStatus demandPropertyListener(
        AudioObjectID inObjectID, UInt32 inNumberAddresses,
        const AudioObjectPropertyAddress* inAddresses, void* inClientData);
};

} // namespace eve
