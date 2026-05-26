// good

// utils.cpp
// Timer and angle utility implementations.
// Original source: robot_ws/src/src/utils.cpp
//
// Only Timer and ang_diff are kept. The original also had:
// - Coordinate conversions (x_to_j, y_to_i, q_to_yaw, etc.)
// - Interpolation functions (trilinear, bilinear)
// - Low-pass filter
// - getCurrentDateTime
// Those are for the Poisson safety field pipeline and not needed here.

#include "utils.h"
#include <cmath>

Timer::Timer(bool print){
    print_ = print;
}

void Timer::start() {
    start_time = std::chrono::steady_clock::now();
}

float Timer::time() {
    end_time = std::chrono::steady_clock::now();
    duration = end_time - start_time;
    float dur_ms = duration.count() * 1.0e3f;
    if (print_)
        std::cout << dur_ms << " ms" << std::endl;
    start_time = end_time;
    return dur_ms;
}

float Timer::time(std::string info) {
    if (print_)
        std::cout << info;
    return time();
}

/* Compute difference between two angles wrapped between [-pi, pi] */
float ang_diff(const float a1, const float a2) {
    float a3 = a1 - a2;
    while (a3 <= -M_PI) a3 += 2.0f * M_PI;
    while (a3 > M_PI)   a3 -= 2.0f * M_PI;
    return a3;
}
