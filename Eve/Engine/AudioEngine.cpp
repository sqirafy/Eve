#include "AudioEngine.hpp"
#include "SPSCRingBuffer.hpp"
#include "AudioCapture.hpp"
#include "AudioOutput.hpp"
#include "InferenceWorker.hpp"
#include "../Audio/CoreMLInference.h"

namespace eve {

AudioEngine::AudioEngine()
    : capture_buffer_(  std::make_unique<SPSCRingBuffer<float>>(kBufCapacity))
    , processed_buffer_(std::make_unique<SPSCRingBuffer<float>>(kBufCapacity))
    , mic_capture_(     std::make_unique<AudioCapture>())
    , worker_(          std::make_unique<InferenceWorker>())
    , blackhole_output_(std::make_unique<AudioOutput>())
{}

AudioEngine::~AudioEngine() {
    stop();
    if (model_context_) {
        eve_model_unload(model_context_);
        model_context_ = nullptr;
    }
}

bool AudioEngine::loadModel(const char* modelPath) {
    if (model_context_) {
        eve_model_unload(model_context_);
        model_context_ = nullptr;
    }
    model_context_ = eve_model_load(modelPath);
    return model_context_ != nullptr;
}

bool AudioEngine::start(AudioDeviceID micDeviceID) {
    if (running_.load(std::memory_order_relaxed)) stop();
    if (!model_context_) {
        fprintf(stderr, "Eve: start failed — model not loaded\n");
        return false;
    }

    blackhole_id_ = findOutputDeviceByName("BlackHole");
    if (blackhole_id_ == kAudioObjectUnknown) {
        fprintf(stderr, "Eve: start failed — BlackHole not found\n");
        return false;
    }

    capture_buffer_->reset();
    processed_buffer_->reset();

    // Wire mic capture → ring buffer → inference worker → ring buffer → output.
    mic_capture_->setOutputBuffer(capture_buffer_.get());
    mic_capture_->setNotifyCallback(
        [](void* ctx) { static_cast<InferenceWorker*>(ctx)->signal(); },
        worker_.get());

    worker_->setInputBuffer(capture_buffer_.get());
    worker_->setOutputBuffer(processed_buffer_.get());
    worker_->setInferenceCallback(eve_model_predict, model_context_);
    worker_->setDemand(false);  // start idle — demand monitor will activate

    blackhole_output_->setInputBuffer(processed_buffer_.get());

    // Start inference worker thread (idles when demand=false, just drains capture).
    worker_->start();

    // Start mic capture.
    if (!mic_capture_->start(micDeviceID, 48000.0)) {
        fprintf(stderr, "Eve: start failed — mic capture start failed\n");
        worker_->stop();
        return false;
    }

    // NOTE: BlackHole output is NOT started yet — demand monitor will start it
    // when an external app begins reading from BlackHole.
    running_.store(true, std::memory_order_release);

    startDemandMonitoring();
    return true;
}

void AudioEngine::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    stopDemandMonitoring();
    mic_capture_->stop();
    worker_->stop();
    blackhole_output_->stop();
    has_demand_.store(false, std::memory_order_release);
    running_.store(false, std::memory_order_release);
}

bool AudioEngine::isRunning() const {
    return running_.load(std::memory_order_acquire);
}

bool AudioEngine::isBlackHoleConnected() const {
    return blackhole_output_ && blackhole_output_->isRunning();
}

void AudioEngine::setPassthrough(bool passthrough) {
    if (worker_) worker_->setPassthrough(passthrough);
}

// ---------------------------------------------------------------------------
// Demand monitoring
// ---------------------------------------------------------------------------

/// Check if BlackHole has active IO in any process.
static bool isDeviceRunningSomewhere(AudioDeviceID deviceID) {
    UInt32 isRunning = 0;
    UInt32 size = sizeof(isRunning);
    AudioObjectPropertyAddress prop = {
        kAudioDevicePropertyDeviceIsRunningSomewhere,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    OSStatus err = AudioObjectGetPropertyData(deviceID, &prop, 0, nullptr, &size, &isRunning);
    return err == noErr && isRunning != 0;
}

void AudioEngine::startDemandMonitoring() {
    if (blackhole_id_ == kAudioObjectUnknown) return;

    // Register a property listener for when any process starts/stops IO on BlackHole.
    AudioObjectPropertyAddress prop = {
        kAudioDevicePropertyDeviceIsRunningSomewhere,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectAddPropertyListener(blackhole_id_, &prop, demandPropertyListener, this);

    // NOTE: Auto-stop via periodic IO proc stop/check/restart is intentionally
    // omitted. BlackHole is a zero-latency loopback with no internal buffering —
    // stopping our output IO proc even briefly causes an audible click in any app
    // reading from BlackHole's input side. The property listener above handles
    // auto-start cleanly. Use the noise suppression toggle to stop inference.
}

void AudioEngine::stopDemandMonitoring() {
    // Remove property listener.
    if (blackhole_id_ != kAudioObjectUnknown) {
        AudioObjectPropertyAddress prop = {
            kAudioDevicePropertyDeviceIsRunningSomewhere,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        AudioObjectRemovePropertyListener(blackhole_id_, &prop, demandPropertyListener, this);
    }

}

void AudioEngine::onDemandStarted() {
    if (has_demand_.load(std::memory_order_acquire)) return;  // already active

    processed_buffer_->reset();
    worker_->setDemand(true);

    if (!blackhole_output_->isRunning()) {
        blackhole_output_->start(blackhole_id_, 48000.0);
    }

    has_demand_.store(true, std::memory_order_release);
}

OSStatus AudioEngine::demandPropertyListener(
    AudioObjectID /*inObjectID*/, UInt32 /*inNumberAddresses*/,
    const AudioObjectPropertyAddress* /*inAddresses*/, void* inClientData)
{
    auto* self = static_cast<AudioEngine*>(inClientData);

    // Property changed — if we don't have demand yet and BlackHole is now active,
    // an external app just opened it. Start inference.
    if (!self->has_demand_.load(std::memory_order_acquire)) {
        if (isDeviceRunningSomewhere(self->blackhole_id_)) {
            // Dispatch to avoid doing heavy work in the property listener callback.
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                if (self->running_.load(std::memory_order_acquire)) {
                    self->onDemandStarted();
                }
            });
        }
    }

    return noErr;
}

} // namespace eve
