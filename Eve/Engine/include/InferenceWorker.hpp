#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <dispatch/dispatch.h>
#include <thread>

namespace eve {

template <typename T>
class SPSCRingBuffer;

/// C-linkage callback type for running inference.
/// Returns 0 on success, non-zero on error.
using InferenceCallback = int (*)(const float* input,   // [kFrameSamples]
                                   float* output,        // [kHopSamples]
                                   void* model_context);

/// Dedicated thread that reads captured audio from the input ring buffer,
/// runs DPDFNet inference, and writes enhanced audio to an output ring buffer
/// (consumed by AudioOutput → BlackHole).
class InferenceWorker {
public:
    /// Model frame = 20ms at 48kHz, hop = 10ms at 48kHz.
    static constexpr size_t kFrameSamples = 960;
    static constexpr size_t kHopSamples   = 480;

    InferenceWorker();
    ~InferenceWorker();

    InferenceWorker(const InferenceWorker&) = delete;
    InferenceWorker& operator=(const InferenceWorker&) = delete;

    /// Configure the worker before starting.
    void setInputBuffer(SPSCRingBuffer<float>* buffer);
    void setOutputBuffer(SPSCRingBuffer<float>* buffer);
    void setInferenceCallback(InferenceCallback callback, void* context);

    /// When true, samples are forwarded to the output buffer without inference.
    void setPassthrough(bool passthrough);

    /// When false, the worker drains captured audio without running CoreML
    /// inference — used to skip processing when no app is reading from BlackHole.
    void setDemand(bool demand);

    /// Signal the worker that new audio samples are available.
    /// Call this from the audio capture callback after pushing to the ring buffer.
    /// Real-time safe (lock-free).
    void signal();

    /// Start the inference thread.
    void start();

    /// Stop the inference thread (blocks until joined).
    void stop();

    bool isRunning() const { return running_.load(std::memory_order_acquire); }

private:
    void threadFunc();

    SPSCRingBuffer<float>* input_buffer_  = nullptr;
    SPSCRingBuffer<float>* output_buffer_ = nullptr;
    InferenceCallback      inference_cb_  = nullptr;
    void* model_context_ = nullptr;

    std::thread thread_;
    dispatch_semaphore_t   semaphore_ = nullptr;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> passthrough_{true};   // start in passthrough; caller sets false for inference
    std::atomic<bool> demand_{false};        // true when a client app is reading from BlackHole

    // Internal frame assembly buffer.
    // We accumulate kFrameSamples (960) for the first frame,
    // then shift by kHopSamples (480) for subsequent frames (50% overlap).
    float frame_buffer_[kFrameSamples] = {};
    float hop_output_[kHopSamples] = {};
    size_t frame_fill_ = 0;
    bool first_frame_ = true;
};

} // namespace eve
