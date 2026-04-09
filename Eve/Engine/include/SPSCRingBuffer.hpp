#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

namespace eve {

/// Lock-free single-producer single-consumer ring buffer.
/// Designed for real-time audio threads: no allocations after construction,
/// no locks, no system calls. Cache-line aligned atomics prevent false sharing.
template <typename T>
class SPSCRingBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SPSCRingBuffer requires trivially copyable types");

public:
    /// Capacity must be a power of two. Rounds up if not.
    explicit SPSCRingBuffer(size_t requested_capacity)
        : mask_(nextPowerOfTwo(requested_capacity) - 1)
        , buffer_(std::make_unique<T[]>(mask_ + 1))
    {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

    ~SPSCRingBuffer() = default;

    // Non-copyable, non-movable (atomics)
    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer(SPSCRingBuffer&&) = delete;
    SPSCRingBuffer& operator=(SPSCRingBuffer&&) = delete;

    /// Push up to `count` elements. Returns number actually written.
    /// Called from producer thread only.
    size_t push(const T* data, size_t count) noexcept {
        const size_t w = write_pos_.load(std::memory_order_relaxed);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        const size_t available = capacity() - (w - r);
        const size_t n = (count < available) ? count : available;

        if (n == 0) return 0;

        const size_t write_idx = w & mask_;
        const size_t first_chunk = capacity() - write_idx;

        if (n <= first_chunk) {
            std::memcpy(buffer_.get() + write_idx, data, n * sizeof(T));
        } else {
            std::memcpy(buffer_.get() + write_idx, data, first_chunk * sizeof(T));
            std::memcpy(buffer_.get(), data + first_chunk, (n - first_chunk) * sizeof(T));
        }

        write_pos_.store(w + n, std::memory_order_release);
        return n;
    }

    /// Pop up to `count` elements. Returns number actually read.
    /// Called from consumer thread only.
    size_t pop(T* data, size_t count) noexcept {
        const size_t r = read_pos_.load(std::memory_order_relaxed);
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t available = w - r;
        const size_t n = (count < available) ? count : available;

        if (n == 0) return 0;

        const size_t read_idx = r & mask_;
        const size_t first_chunk = capacity() - read_idx;

        if (n <= first_chunk) {
            std::memcpy(data, buffer_.get() + read_idx, n * sizeof(T));
        } else {
            std::memcpy(data, buffer_.get() + read_idx, first_chunk * sizeof(T));
            std::memcpy(data + first_chunk, buffer_.get(), (n - first_chunk) * sizeof(T));
        }

        read_pos_.store(r + n, std::memory_order_release);
        return n;
    }

    /// Number of elements available to read.
    size_t available_read() const noexcept {
        const size_t w = write_pos_.load(std::memory_order_acquire);
        const size_t r = read_pos_.load(std::memory_order_relaxed);
        return w - r;
    }

    /// Number of elements that can be written.
    size_t available_write() const noexcept {
        const size_t w = write_pos_.load(std::memory_order_relaxed);
        const size_t r = read_pos_.load(std::memory_order_acquire);
        return capacity() - (w - r);
    }

    /// Total capacity of the buffer.
    size_t capacity() const noexcept { return mask_ + 1; }

    /// Reset both positions to zero. NOT thread-safe — call only when idle.
    void reset() noexcept {
        write_pos_.store(0, std::memory_order_relaxed);
        read_pos_.store(0, std::memory_order_relaxed);
    }

private:
    static size_t nextPowerOfTwo(size_t v) {
        if (v == 0) return 1;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return v + 1;
    }

    const size_t mask_;
    std::unique_ptr<T[]> buffer_;

    // Cache-line aligned to prevent false sharing between producer and consumer.
    // ARM64 (Apple Silicon) has 128-byte cache lines; x86_64 has 64-byte.
#if defined(__aarch64__)
    alignas(128) std::atomic<size_t> write_pos_;
    alignas(128) std::atomic<size_t> read_pos_;
#else
    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
#endif
};

} // namespace eve
