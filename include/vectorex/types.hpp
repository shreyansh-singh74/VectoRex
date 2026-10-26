#pragma once

#include <cstddef>

namespace vectorex {

struct SearchResult {
  std::size_t id;
  float distance;
};

enum class DistanceMetric {
  SquaredL2,
  Cosine,
  InnerProduct
};

} // namespace vectorex
