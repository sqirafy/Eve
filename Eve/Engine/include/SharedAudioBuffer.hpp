#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace eve {

/// Shared memory name used by both the app and the AudioServerPlugin driver.
static constexpr const char* kSharedMemoryName = "/eve_audio_bridge";

/// Ring buffer capacity in samples (~2 seconds at 48kHz).
static constexpr uint64_t kSharedBufferCapacity = 131072; // 2^17, power of two
static constexpr uint64_t kSharedBufferMask = kSharedBufferCapacity - 1;

/// Layout of the POSIX shared memory segment.
/// Both the main app (writer) and the AudioServerPlugin (reader) map this.
struct SharedAudioBuffer {
    alignas(64) std::atomic<uint64_t> write_pos;
    alignas(64) std::atomic<uint64_t> read_pos;
    alignas(64) std::atomic<int32_t>  active_clients; // incremented on StartIO, decremented on StopIO
    alignas(64) float samples[kSharedBufferCapacity];

    void reset() {
        write_pos.store(0, std::memory_order_relaxed);
        read_pos.store(0, std::memory_order_relaxed);
        active_clients.store(0, std::memory_order_relaxed);
        std::memset(samples, 0, sizeof(samples));
    }

    /// Push samples into the shared ring buffer (called by main app).
    uint64_t push(const float* data, uint64_t count) {
        const uint64_t w = write_pos.load(std::memory_order_relaxed);
        const uint64_t r = read_pos.load(std::memory_order_acquire);
        const uint64_t available = kSharedBufferCapacity - (w - r);
        const uint64_t n = (count < available) ? count : available;

        if (n == 0) return 0;

        const uint64_t write_idx = w & kSharedBufferMask;
        const uint64_t first_chunk = kSharedBufferCapacity - write_idx;

        if (n <= first_chunk) {
            std::memcpy(samples + write_idx, data, n * sizeof(float));
        } else {
            std::memcpy(samples + write_idx, data, first_chunk * sizeof(float));
            std::memcpy(samples, data + first_chunk, (n - first_chunk) * sizeof(float));
        }

        write_pos.store(w + n, std::memory_order_release);
        return n;
    }

    /// Pop samples from the shared ring buffer (called by AudioServerPlugin).
    uint64_t pop(float* data, uint64_t count) {
        const uint64_t r = read_pos.load(std::memory_order_relaxed);
        const uint64_t w = write_pos.load(std::memory_order_acquire);
        const uint64_t available = w - r;
        const uint64_t n = (count < available) ? count : available;

        if (n == 0) return 0;

        const uint64_t read_idx = r & kSharedBufferMask;
        const uint64_t first_chunk = kSharedBufferCapacity - read_idx;

        if (n <= first_chunk) {
            std::memcpy(data, samples + read_idx, n * sizeof(float));
        } else {
            std::memcpy(data, samples + read_idx, first_chunk * sizeof(float));
            std::memcpy(data + first_chunk, samples, (n - first_chunk) * sizeof(float));
        }

        read_pos.store(r + n, std::memory_order_release);
        return n;
    }

    uint64_t availableRead() const {
        const uint64_t w = write_pos.load(std::memory_order_acquire);
        const uint64_t r = read_pos.load(std::memory_order_relaxed);
        return w - r;
    }
};

/// Helper to create/open the shared memory segment.
/// Returns nullptr on failure.
inline SharedAudioBuffer* openSharedAudioBuffer(bool create) {
    int flags = O_RDWR;
    if (create) flags |= O_CREAT;

    int fd = shm_open(kSharedMemoryName, flags, 0666);
    if (fd < 0) return nullptr;

    if (create) {
        if (ftruncate(fd, sizeof(SharedAudioBuffer)) != 0) {
            close(fd);
            return nullptr;
        }
    }

    void* ptr = mmap(nullptr, sizeof(SharedAudioBuffer),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) return nullptr;

    auto* buf = static_cast<SharedAudioBuffer*>(ptr);
    if (create) {
        buf->reset();
    }

    return buf;
}

inline void closeSharedAudioBuffer(SharedAudioBuffer* buf) {
    if (buf) {
        munmap(buf, sizeof(SharedAudioBuffer));
    }
}

inline void unlinkSharedAudioBuffer() {
    shm_unlink(kSharedMemoryName);
}

} // namespace eve
