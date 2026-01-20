#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <iostream>

typedef std::chrono::high_resolution_clock Clock;

class Timer {
private:
    Clock::time_point point;

public:
    Timer() { this->restart(); }
    
    void restart() {
        point = Clock::now();
    }

    void printMicro(std::string message = "Elapsed") {
        Clock::duration elapsed = this->timeElapsed();
        std::cout << message << ": " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "\n";
    }

    void printMilli(std::string message = "Elapsed") {
        Clock::duration elapsed = this->timeElapsed();
        std::cout << message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "\n";
    }

    static void printMilli(Clock::duration elapsed, std::string message = "Elapsed") {
        std::cout << message << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "\n";
    }

    Clock::duration timeElapsed() {
        return Clock::now() - point;
    }
};

// Store elapsed times. "capture" captures the time from now to the last time capture was called
// Using arrays to avoid overhead of vectors
template<int MAX_SIZE = 100>
class TimerCaptures {
    private:
    long long elapsedTimes[MAX_SIZE];
    unsigned int currSize = 0;
    Timer timer;

    public:
    void capture() {
        if (currSize == MAX_SIZE) return;
        Clock::duration timeElapsed = timer.timeElapsed();

        long long timeElapsedCount = std::chrono::duration_cast<std::chrono::milliseconds>(timeElapsed).count();
        elapsedTimes[currSize] = timeElapsedCount;
        currSize++;

        timer.restart();
    }

    long long getAverageElapsed() {
        long long sum = 0.0f;
        for (int i=0; i<currSize; i++) {
            sum += elapsedTimes[i];
        }
        return sum / double(currSize);
    }

    void printAverageElapsed(std::string message = "Elapsed") {
        long long elapsed = this->getAverageElapsed();
        std::cout << message << ": " << elapsed << "\n";
    }
};