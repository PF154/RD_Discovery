#pragma once

#include "pattern_detection.cuh"
#include "thread_safe_queue.h"
#include <thread>
#include <atomic>
#include <vector>

/**
 * Asynchronous pattern detector using producer-consumer pattern.
 *
 * Main thread submits work via submit_work() (non-blocking)
 * Worker thread processes work in background (blocks on GPU)
 * Main thread retrieves results via try_get_result() (non-blocking)
 *
 * This keeps the UI thread responsive while GPU work happens in background.
 */
class AsyncPatternDetector {
public:
    /**
     * Work item: Input to pattern detection pipeline
     */
    struct WorkItem {
        std::vector<ParamSet> params;  // Batch of parameters to test
        int request_id;                 // Identifier for this request
    };

    /**
     * Result: Output from pattern detection pipeline
     */
    struct Result {
        std::vector<PatternResult> patterns;  // Detected patterns
        int request_id;                        // Matches WorkItem::request_id
    };

private:
    // Communication queues (main <-> worker thread)
    ThreadSafeQueue<WorkItem> m_work_queue;    // Main -> Worker
    ThreadSafeQueue<Result> m_result_queue;    // Worker -> Main

    // Worker thread management
    std::thread m_worker_thread;               // Background thread
    std::atomic<bool> m_shutdown;              // Shutdown signal

    /**
     * Worker thread function - runs in background
     * Continuously pops work, runs pattern detection, pushes results
     */
    void worker_loop();

public:
    /**
     * Constructor: Launches worker thread
     */
    AsyncPatternDetector();

    /**
     * Destructor: Stops worker thread and waits for it to finish
     * All pending work in queues will be lost
     */
    ~AsyncPatternDetector();

    // Non-copyable, non-movable (manages thread)
    AsyncPatternDetector(const AsyncPatternDetector&) = delete;
    AsyncPatternDetector& operator=(const AsyncPatternDetector&) = delete;
    AsyncPatternDetector(AsyncPatternDetector&&) = delete;
    AsyncPatternDetector& operator=(AsyncPatternDetector&&) = delete;

    /**
     * Submit work for pattern detection (non-blocking)
     * Called by main thread
     *
     * Caller can move params if not needed after:
     *   submit_work(std::move(params), id);
     *
     * @param params Batch of parameters to test (taken by value, can be moved)
     * @param request_id Identifier for tracking this request
     */
    void submit_work(std::vector<ParamSet> params, int request_id);

    /**
     * Try to get one result (non-blocking)
     * Called by main thread
     *
     * @param result Output parameter - filled if result available
     * @return true if result was retrieved, false if no results available
     */
    bool try_get_result(Result& result);

    /**
     * Try to get multiple results in batch (non-blocking)
     * More efficient than calling try_get_result() in a loop
     *
     * @param results Output vector - results will be appended
     * @param max_count Maximum number of results to retrieve (0 = unlimited)
     * @return Number of results actually retrieved
     */
    size_t try_get_results(std::vector<Result>& results, size_t max_count = 0);

    /**
     * Get number of pending work items
     * Thread-safe
     */
    size_t pending_work_count() const;

    /**
     * Get number of pending results
     * Thread-safe
     */
    size_t pending_results_count() const;
};
