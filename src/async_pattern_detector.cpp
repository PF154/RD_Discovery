#include "async_pattern_detector.h"
#include "pattern_detection.cuh"
#include <thread>

AsyncPatternDetector::AsyncPatternDetector()
{
    shutdown_ = false;

    worker_thread_ = std::thread(&AsyncPatternDetector::worker_loop, this);
}

AsyncPatternDetector::~AsyncPatternDetector()
{
    shutdown_ = true;

    // Push dummy item to wake worker
    work_queue_.push(WorkItem{});

    worker_thread_.join();
}

void AsyncPatternDetector::worker_loop()
{
    while(!shutdown_)
    {
        WorkItem item = work_queue_.pop();

        if (shutdown_) break;

        std::vector<PatternResult> results = detect_patterns_batch(item.params);

        Result result = {std::move(results), item.request_id};

        result_queue_.push(std::move(result));
    }
}

void AsyncPatternDetector::submit_work(std::vector<ParamSet> params, int request_id)
{
    if (work_queue_.size() > 10) return;

    WorkItem item = {std::move(params), request_id};

    work_queue_.push(std::move(item));
}


bool AsyncPatternDetector::try_get_result(Result& result)
{
    return result_queue_.try_pop(result);
}

size_t AsyncPatternDetector::try_get_results(std::vector<Result>& results, size_t max_count)
{
    return result_queue_.try_pop_batch(results, max_count);
}

size_t AsyncPatternDetector::pending_work_count() const
{
    return work_queue_.size();
}

size_t AsyncPatternDetector::pending_results_count() const
{
    return result_queue_.size();
}
