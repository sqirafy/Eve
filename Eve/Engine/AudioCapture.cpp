#include "AudioCapture.hpp"
#include "SPSCRingBuffer.hpp"
#include <cstring>
#include <algorithm>

namespace eve {

AudioCapture::AudioCapture() = default;

AudioCapture::~AudioCapture() {
    stop();
}

void AudioCapture::setOutputBuffer(SPSCRingBuffer<float>* buffer) {
    output_buffer_ = buffer;
}

void AudioCapture::setNotifyCallback(NotifyCallback cb, void* ctx) {
    notify_cb_  = cb;
    notify_ctx_ = ctx;
}

bool AudioCapture::configureDevice(AudioDeviceID deviceID, Float64 targetSampleRate) {
    // Query the device's current nominal sample rate.
    Float64 currentRate = 0;
    UInt32 size = sizeof(currentRate);
    AudioObjectPropertyAddress prop = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };

    OSStatus err = AudioObjectGetPropertyData(deviceID, &prop, 0, nullptr, &size, &currentRate);
    if (err != noErr) return false;

    actual_sample_rate_ = currentRate;

    // Try to set the device to the target sample rate if different.
    if (currentRate != targetSampleRate) {
        Float64 newRate = targetSampleRate;
        err = AudioObjectSetPropertyData(deviceID, &prop, 0, nullptr, sizeof(newRate), &newRate);
        if (err == noErr) {
            actual_sample_rate_ = targetSampleRate;
        }
        // If setting fails, we'll use a sample rate converter.
    }

    // Query number of input channels.
    AudioObjectPropertyAddress streamProp = {
        kAudioDevicePropertyStreamConfiguration,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };

    size = 0;
    err = AudioObjectGetPropertyDataSize(deviceID, &streamProp, 0, nullptr, &size);
    if (err != noErr) return false;

    auto bufferListStorage = std::make_unique<uint8_t[]>(size);
    auto* bufferList = reinterpret_cast<AudioBufferList*>(bufferListStorage.get());
    err = AudioObjectGetPropertyData(deviceID, &streamProp, 0, nullptr, &size, bufferList);
    if (err != noErr) return false;

    device_channels_ = 0;
    for (UInt32 i = 0; i < bufferList->mNumberBuffers; i++) {
        device_channels_ += bufferList->mBuffers[i].mNumberChannels;
    }
    if (device_channels_ == 0) device_channels_ = 1;

    // Set buffer size for low latency (256 frames).
    UInt32 bufferFrameSize = 256;
    AudioObjectPropertyAddress bufSizeProp = {
        kAudioDevicePropertyBufferFrameSize,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    AudioObjectSetPropertyData(deviceID, &bufSizeProp, 0, nullptr,
                               sizeof(bufferFrameSize), &bufferFrameSize);

    // Set up sample rate converter if needed.
    if (converter_) {
        AudioConverterDispose(converter_);
        converter_ = nullptr;
    }

    if (actual_sample_rate_ != targetSampleRate) {
        AudioStreamBasicDescription srcDesc = {};
        srcDesc.mSampleRate = actual_sample_rate_;
        srcDesc.mFormatID = kAudioFormatLinearPCM;
        srcDesc.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        srcDesc.mBitsPerChannel = 32;
        srcDesc.mChannelsPerFrame = 1;
        srcDesc.mFramesPerPacket = 1;
        srcDesc.mBytesPerFrame = 4;
        srcDesc.mBytesPerPacket = 4;

        AudioStreamBasicDescription dstDesc = srcDesc;
        dstDesc.mSampleRate = targetSampleRate;

        err = AudioConverterNew(&srcDesc, &dstDesc, &converter_);
        if (err != noErr) {
            converter_ = nullptr;
            // Fall back to using device rate without conversion.
        }
    }

    return true;
}

bool AudioCapture::start(AudioDeviceID deviceID, Float64 targetSampleRate) {
    if (running_.load(std::memory_order_relaxed)) {
        stop();
    }

    if (!output_buffer_) return false;

    target_sample_rate_ = targetSampleRate;

    if (!configureDevice(deviceID, targetSampleRate)) {
        return false;
    }

    device_id_ = deviceID;

    // Register the IO proc.
    OSStatus err = AudioDeviceCreateIOProcID(device_id_, ioProc, this, &io_proc_id_);
    if (err != noErr) return false;

    // Start the device.
    err = AudioDeviceStart(device_id_, io_proc_id_);
    if (err != noErr) {
        AudioDeviceDestroyIOProcID(device_id_, io_proc_id_);
        io_proc_id_ = nullptr;
        return false;
    }

    running_.store(true, std::memory_order_release);
    return true;
}

void AudioCapture::stop() {
    if (!running_.load(std::memory_order_acquire)) return;

    running_.store(false, std::memory_order_release);

    if (device_id_ != kAudioObjectUnknown && io_proc_id_) {
        AudioDeviceStop(device_id_, io_proc_id_);
        AudioDeviceDestroyIOProcID(device_id_, io_proc_id_);
    }

    io_proc_id_ = nullptr;
    device_id_ = kAudioObjectUnknown;

    if (converter_) {
        AudioConverterDispose(converter_);
        converter_ = nullptr;
    }
}

bool AudioCapture::isRunning() const {
    return running_.load(std::memory_order_acquire);
}

// Real-time audio callback — NO allocations, NO locks, NO ObjC messaging.
OSStatus AudioCapture::ioProc(AudioDeviceID /*device*/,
                               const AudioTimeStamp* /*now*/,
                               const AudioBufferList* inputData,
                               const AudioTimeStamp* /*inputTime*/,
                               AudioBufferList* /*outputData*/,
                               const AudioTimeStamp* /*outputTime*/,
                               void* clientData)
{
    auto* self = static_cast<AudioCapture*>(clientData);
    if (!self->output_buffer_ || !inputData || inputData->mNumberBuffers == 0) {
        return noErr;
    }

    const AudioBuffer& buf = inputData->mBuffers[0];
    const auto* src = static_cast<const float*>(buf.mData);
    const UInt32 frameCount = buf.mDataByteSize / (buf.mNumberChannels * sizeof(float));

    if (!src || frameCount == 0) return noErr;

    // If multi-channel, extract first channel (mono downmix).
    // Use the stack-based convert buffer for intermediate storage.
    const UInt32 channels = buf.mNumberChannels;
    const float* monoData = src;
    float monoBuffer[2048]; // stack buffer, safe for real-time at typical frame sizes

    if (channels > 1 && frameCount <= 2048) {
        for (UInt32 i = 0; i < frameCount; i++) {
            monoBuffer[i] = src[i * channels]; // take first channel
        }
        monoData = monoBuffer;
    }

    // Push mono samples into the ring buffer.
    self->output_buffer_->push(monoData, frameCount);

    // Signal the inference worker that new samples are available.
    if (self->notify_cb_) self->notify_cb_(self->notify_ctx_);

    return noErr;
}

} // namespace eve
