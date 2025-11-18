#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;  // For consumers waiting for data
    std::condition_variable not_full_;   // For producers waiting for space
    const size_t max_size_;
    bool shutdown_;

public:
    ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    ~ThreadSafeQueue() {
        shutdown_ = true;
        not_empty_.notify_all();  // Wake all waiting consumers
        not_full_.notify_all();   // Wake all waiting producers
    }

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /**
     * Producer: Add item to queue
     * Thread-safe, BLOCKING if queue is full
     *
     * @param item The item to add (will be moved if rvalue)
     */
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has space (blocks if full)
        not_full_.wait(lock, [this] { return queue_.size() < max_size_; });

        queue_.push(std::move(item));

        lock.unlock();
        not_empty_.notify_one();
    }

    /**
     * Consumer: Get and remove item from queue
     * Thread-safe, BLOCKING - waits if queue is empty
     *
     * @return The front item from the queue
     */
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has data
        not_empty_.wait(lock, [this] { return !queue_.empty(); });

        T item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    /**
     * Consumer: Try to get item without blocking
     * Thread-safe, NON-BLOCKING
     *
     * This is what the main thread calls - never blocks
     *
     * @param item Output parameter - set to the front item if available
     * @return true if item was retrieved, false if queue was empty
     */
    bool try_pop(T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (queue_.empty()) {
                return false;
            }

            item = std::move(queue_.front());
            queue_.pop();
        }

        not_full_.notify_one();
        return true;
    }

    /**
     * Consumer: Try to pop multiple items in one batch
     * Thread-safe, NON-BLOCKING
     *
     * Much more efficient than calling try_pop() in a loop:
     * - Single lock acquisition instead of N
     * - Better cache locality
     * - Reduced mutex contention
     *
     * @param items Output vector - items will be appended to this
     * @param max_count Maximum number of items to pop (0 = unlimited)
     * @return Number of items actually popped
     */
    size_t try_pop_batch(std::vector<T>& items, size_t max_count = 0) {
        size_t count;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Determine how many items to pop
            count = queue_.size();
            if (max_count > 0 && count > max_count) {
                count = max_count;
            }

            if (count == 0) {
                return 0;
            }

            // Reserve space to avoid reallocations
            items.reserve(items.size() + count);

            // Pop all items in batch
            for (size_t i = 0; i < count; i++) {
                items.push_back(std::move(queue_.front()));
                queue_.pop();
            }
        }  // Lock released here

        // Wake up producers if we freed space
        // Could use notify_all() if count is large
        if (count > 0) {
            not_full_.notify_one();
        }

        return count;
    }

    /**
     * Check if queue is empty
     * Thread-safe
     *
     * Note: Result may be stale immediately after return due to concurrent operations
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * Get queue size
     * Thread-safe
     *
     * Note: Result may be stale immediately after return due to concurrent operations
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};
