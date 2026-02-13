#pragma once

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>
#include <functional>

class ThreadPool {
private:
    std::vector<std::thread> threads;
    
    std::queue<std::function<void()>> jobs;
    std::mutex queueMutex;

    std::condition_variable notify;
    bool stop = false;

    size_t numThreads;

public:
    ThreadPool(size_t _numThreads = std::thread::hardware_concurrency()): numThreads(_numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        notify.wait(lock, [this] {
                            bool isThereAJob = !jobs.empty();
                            return isThereAJob || stop;
                        });

                        if (stop && jobs.empty()) {
                            return;
                        }

                        job = std::move(jobs.front());
                        jobs.pop();
                    }

                    job();
                }
            });
        }
    }

    void enqueue(std::function<void()> job) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            jobs.emplace(std::move(job));
        }
        notify.notify_one();
    }

    size_t getNumThreads() {
        return numThreads;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        notify.notify_all();

        for (auto& thread : threads) {
            thread.join();
        }
    }
};