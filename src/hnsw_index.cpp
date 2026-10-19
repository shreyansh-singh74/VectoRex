#include "vectorex/hnsw_index.hpp"
#include "vectorex/distance.hpp"
#include "vectorex/vector.hpp"
#include "vectorex/vector_store.hpp"
#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
namespace vectorex {
HnswIndex::Node::Node(std::size_t node_id, Vector node_vector, int level)
    : id(node_id), vector(std::move(node_vector)),
      neighbours(static_cast<std::size_t>(level) + 1) {}

HnswIndex::HnswIndex(std::size_t dimension, std::size_t M,
                     std::size_t ef_construction, std::size_t ef_search)
    : dimension_(dimension), M_(M), ef_construction_(ef_construction),
      ef_search_(ef_search), rng_(std::random_device{}()) {
  if (dimension_ == 0) {
    throw std::invalid_argument("HNSW dimension must be greater than zero");
  }

  if (M_ < 2) {
    throw std::invalid_argument("HNSW M must be at least 2");
  }

  if (ef_construction_ < M_) {
    throw std::invalid_argument("ef_construction must be at least M");
  }

  if (ef_search == 0) {
    throw std::invalid_argument("ef_search must be greater than zero");
  }
}

float HnswIndex::calculate_distance(const Vector &a, const Vector &b,
                                    DistanceMetric metric) const {
  switch (metric) {
  case DistanceMetric::SquaredL2:
    return squared_l2_distance(a, b);

  case DistanceMetric::Cosine:
    return 1.0f - cosine_similarity(a, b);

  case DistanceMetric::InnerProduct:
    return -inner_product(a, b);
  }
  throw std::invalid_argument("Unsupported distnace metric");
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

  int level = 0;

  while (distribution(rng_) < (1.0f / static_cast<float>(M_))) {
    ++level;
  }

  return level;
}

bool HnswIndex::constains(std::size_t id) const {
  return id_to_index_.find(id) != id_to_index_.end();
}

std::size_t HnswIndex::size() const { return nodes_.size(); }

std::size_t HnswIndex::dimension() const { return dimension_; }

void HnswIndex::set_ef_search(std::size_t ef_search) {
  if (ef_search == 0) {
    throw std::invalid_argument("ef_search must be greater than zero");
  }
  ef_search_ = ef_search;
}

std::size_t HnswIndex::greedy_search_layer(const Vector &query,
                                           std::size_t entry, int level,
                                           DistanceMetric metric) const {
  std::size_t current = entry;
  float current_distance =
      calculate_distance(query, nodes_[current].vector, metric);

  bool improved = true;

  while (improved) {
    improved = false;

    for (std::size_t neighbour : nodes_[current].neighbours[level]) {
      float distance =
          calculate_distance(query, nodes_[neighbour].vector, metric);

      if (distance < current_distance) {
        current = neighbour;
        current_distance = distance;
        improved = true;
      }
    }
  }
  return current;
}

std::vector<std::size_t>
HnswIndex::search_layer(const Vector &query,
                        const std::vector<std::size_t> &entries, std::size_t ef,
                        int level, DistanceMetric metric) const {
  using Candidate = std::pair<float, std::size_t>;

  std::priority_queue<Candidate, std::vector<Candidate>,
                      std::greater<Candidate>>
      candidates;

  std::priority_queue<Candidate> best;

  std::vector<bool> visited(nodes_.size(), false);

  for (std::size_t entry : entries) {
    float distance = calculate_distance(query, nodes_[entry].vector, metric);

    candidates.push({distance, entry});
    best.push({distance, entry});
    visited[entry] = true;
  }

  while (!candidates.empty()) {
    Candidate current = candidates.top();

    if (best.size() >= ef && current.first > best.top().first) {
      break;
    }

    candidates.pop();

    std::size_t current_index = current.second;

    for (std::size_t neighbour : nodes_[current_index].neighbours[level]) {
      if (visited[neighbour]) {
        continue;
      }

      visited[neighbour] = true;

      float distance =
          calculate_distance(query, nodes_[neighbour].vector, metric);

      if (best.size() < ef || distance < best.top().first) {
        candidates.push({distance, neighbour});
        best.push({distance, neighbour});

        if (best.size() > ef) {
          best.pop();
        }
      }
    }
  }

  std::vector<Candidate> ordered;

  while (!best.empty()) {
    ordered.push_back(best.top());
    best.pop();
  }

  std::sort(ordered.begin(), ordered.end());

  std::vector<std::size_t> result;
  result.reserve(ordered.size());

  for (const Candidate &candidate : ordered) {
    result.push_back(candidate.second);
  }

  return result;
}

std::vector<std::size_t>
HnswIndex::select_neighbours(const Vector &query,
                             const std::vector<std::size_t> &candidates,
                             std::size_t count, DistanceMetric metric) const {

  std::vector<std::pair<float, std::size_t>> ranked;
  ranked.reserve(candidates.size());

  for (std::size_t index : candidates) {
      ranked.push_back({
          calculate_distance(query, nodes_[index].vector, metric),
          index
      });
  }

  std::sort(ranked.begin(),ranked.end());

  std::vector<std::size_t> result;
  result.reserve(std::min(count,ranked.size()));

  for (std::size_t i=0;i<std::min(count,ranked.size());i++) {
      result.push_back(ranked[i].second);
  }
  return result;
}

} // namespace vectorex
