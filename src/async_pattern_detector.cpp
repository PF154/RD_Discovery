#include "async_pattern_detector.h"
#include "pattern_detection.cuh"
#include <thread>

AsyncPatternDetector::AsyncPatternDetector()
{
    m_shutdown = false;

    m_worker_thread = std::thread(&AsyncPatternDetector::worker_loop, this);
}

AsyncPatternDetector::~AsyncPatternDetector()
{
    m_shutdown = true;

    // Push dummy item to wake worker
    m_work_queue.push(WorkItem{});

    m_worker_thread.join();
}

void AsyncPatternDetector::worker_loop()
{
    while(!m_shutdown)
    {
        WorkItem item = m_work_queue.pop();

        if (m_shutdown) break;

        std::vector<PatternResult> results = detect_patterns_batch(item.params);

        Result result = {std::move(results), item.request_id};

        m_result_queue.push(std::move(result));
    }
}

void AsyncPatternDetector::submit_work(std::vector<ParamSet> params, int request_id)
{
    if (m_work_queue.size() > 10) return;

    WorkItem item = {std::move(params), request_id};

    m_work_queue.push(std::move(item));
}


bool AsyncPatternDetector::try_get_result(Result& result)
{
    return m_result_queue.try_pop(result);
}

size_t AsyncPatternDetector::try_get_results(std::vector<Result>& results, size_t max_count)
{
    return m_result_queue.try_pop_batch(results, max_count);
}

size_t AsyncPatternDetector::pending_work_count() const
{
    return m_work_queue.size();
}

size_t AsyncPatternDetector::pending_results_count() const
{
    return m_result_queue.size();
}
