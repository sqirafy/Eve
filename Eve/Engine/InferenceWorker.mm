#include "InferenceWorker.hpp"
#include "SPSCRingBuffer.hpp"
#include <cstring>
#include <thread>
#import <Foundation/Foundation.h>

namespace eve {

InferenceWorker::InferenceWorker()
    : semaphore_(dispatch_semaphore_create(0))
{}

InferenceWorker::~InferenceWorker() {
    stop();
    // semaphore_ is a dispatch_semaphore_t; under ARC it is released automatically.
    semaphore_ = nullptr;
}

void InferenceWorker::setInputBuffer(SPSCRingBuffer<float>* buffer) {
    input_buffer_ = buffer;
}

void InferenceWorker::setOutputBuffer(SPSCRingBuffer<float>* buffer) {
    output_buffer_ = buffer;
}

void InferenceWorker::setInferenceCallback(InferenceCallback callback, void* context) {
    inference_cb_  = callback;
    model_context_ = context;
}

void InferenceWorker::setPassthrough(bool passthrough) {
    passthrough_.store(passthrough, std::memory_order_release);
}

void InferenceWorker::setDemand(bool demand) {
    demand_.store(demand, std::memory_order_release);
}

void InferenceWorker::signal() {
    dispatch_semaphore_signal(semaphore_);
}

void InferenceWorker::start() {
    if (running_.load(std::memory_order_relaxed)) return;
    if (!input_buffer_ || !inference_cb_) return;

    should_stop_.store(false, std::memory_order_relaxed);
    frame_fill_  = 0;
    first_frame_ = true;
    std::memset(frame_buffer_, 0, sizeof(frame_buffer_));

    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&InferenceWorker::threadFunc, this);
}

void InferenceWorker::stop() {
    if (!running_.load(std::memory_order_acquire)) return;

    should_stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    running_.store(false, std::memory_order_release);
}

void InferenceWorker::threadFunc() {
    // Single autoreleasepool for the thread's lifetime. CoreML's ObjC objects
    // (MLFeatureProvider, MLMultiArray) are short-lived and will drain here on
    // thread exit. This avoids the overhead of creating a pool on every predict call.
    @autoreleasepool {

    size_t hop_fill = 0;

    // Pre-computed 50ms timeout for semaphore wait — allows periodic check of should_stop_.
    // Computed once and reused; dispatch_time(DISPATCH_TIME_NOW, ...) must be called fresh
    // each wait to get a relative-to-now deadline.
    static constexpr int64_t kTimeoutNs = 50 * static_cast<int64_t>(NSEC_PER_MSEC);

    while (!should_stop_.load(std::memory_order_acquire)) {
        // Block until the capture callback signals new samples (or 50ms timeout
        // so we can check should_stop_ periodically).
        dispatch_semaphore_wait(semaphore_,
                                dispatch_time(DISPATCH_TIME_NOW, kTimeoutNs));

        // Drain all available samples — the semaphore may have been signaled
        // multiple times while we were processing, so loop until empty.
        for (;;) {
            if (should_stop_.load(std::memory_order_acquire)) return;

            // No demand: drain captured audio to prevent ring buffer overflow,
            // but skip inference entirely (saves CPU when no app reads BlackHole).
            if (!demand_.load(std::memory_order_acquire)) {
                while (input_buffer_->pop(hop_accum_, kHopSamples) > 0) {}
                frame_fill_  = 0;
                first_frame_ = true;
                break;  // back to waiting on semaphore
            }

            // Passthrough: forward samples directly without inference.
            if (passthrough_.load(std::memory_order_acquire)) {
                const size_t got = input_buffer_->pop(hop_accum_, kHopSamples);
                if (got == 0) break;  // no more data — back to waiting
                if (output_buffer_) output_buffer_->push(hop_accum_, got);
                // Reset frame state so inference resumes cleanly after passthrough.
                frame_fill_  = 0;
                first_frame_ = true;
                continue;
            }

            if (first_frame_) {
                // Accumulate a full kFrameSamples window before first inference.
                const size_t needed = kFrameSamples - frame_fill_;
                const size_t got    = input_buffer_->pop(frame_buffer_ + frame_fill_, needed);
                if (got == 0) break;  // no more data — back to waiting
                frame_fill_ += got;

                if (frame_fill_ >= kFrameSamples) {
                    if (inference_cb_(frame_buffer_, hop_output_, model_context_) == 0) {
                        if (output_buffer_) output_buffer_->push(hop_output_, kHopSamples);
                    }
                    first_frame_ = false;
                    hop_fill     = 0;
                }
            } else {
                // Collect one hop of new samples, then slide the window.
                const size_t needed = kHopSamples - hop_fill;
                const size_t got    = input_buffer_->pop(hop_accum_ + hop_fill, needed);
                if (got == 0) break;  // no more data — back to waiting
                hop_fill += got;

                if (hop_fill >= kHopSamples) {
                    std::memmove(frame_buffer_,
                                 frame_buffer_ + kHopSamples,
                                 (kFrameSamples - kHopSamples) * sizeof(float));
                    std::memcpy(frame_buffer_ + (kFrameSamples - kHopSamples),
                                hop_accum_,
                                kHopSamples * sizeof(float));

                    if (inference_cb_(frame_buffer_, hop_output_, model_context_) == 0) {
                        if (output_buffer_) output_buffer_->push(hop_output_, kHopSamples);
                    }
                    hop_fill = 0;
                }
            }
        }
    }

    } // @autoreleasepool
}

} // namespace eve
