// good

// utils.h
// Timer utility and angle helper functions.
// Original source: robot_ws/src/include/poisson/utils.h
//
// Only the Timer class and ang_diff are needed for walking + LiDAR.
// The original also had coordinate conversion functions (x_to_j, y_to_i, etc.),
// interpolation, and low_pass filter — those are for the safety field pipeline.
#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    Timer(bool print);
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::chrono::duration<float> duration;

    void start();
    float time();
    float time(std::string info);

    bool print_;
};

float ang_diff(const float a1, const float a2);
