/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once
#include <chrono> //NOLINT
#include <vector>
#include <numeric>

namespace vacancy {

template <typename T = double>
class Timer {
  std::chrono::system_clock::time_point start_t_, end_t_;
  T elapsed_msec_{-1};
  size_t history_num_{30};
  std::vector<T> history_;

 public:
  Timer() {}
  ~Timer() {}
  explicit Timer(size_t history_num) : history_num_(history_num) {}

  std::chrono::system_clock::time_point start_t() const { return start_t_; }
  std::chrono::system_clock::time_point end_t() const { return end_t_; }

  void Start() { start_t_ = std::chrono::system_clock::now(); }
  void End() {
    end_t_ = std::chrono::system_clock::now();
    elapsed_msec_ = static_cast<T>(
        std::chrono::duration_cast<std::chrono::microseconds>(end_t_ - start_t_)
            .count() *
        0.001);

    history_.push_back(elapsed_msec_);
    if (history_num_ < history_.size()) {
      history_.erase(history_.begin());
    }
  }
  T elapsed_msec() const { return elapsed_msec_; }
  T average_msec() const {
    return static_cast<T>(std::accumulate(history_.begin(), history_.end(), 0) /
                          history_.size());
  }
};

}  // namespace vacancy
