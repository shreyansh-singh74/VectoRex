#pragma once

#include "vectorex/vector.hpp"
#include "vectorex/vector_store.hpp"
#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>

namespace vectorex {
class HnswIndex {
private:
  struct Node {
    std::size_t id;
    Vector vector;
    std::vector<std::vector<std::size_t>> neighbours;

    Node(std::size_t node_id, Vector node_vector, int level);
  };

  std::vector<Node> nodes_;
  std::unordered_map<std::size_t, std::size_t> id_to_index_;

  std::size_t dimension_;
  std::size_t M_;
  std::size_t ef_construction_;
  std::size_t ef_search_;

  int max_level_ = -1;
  std::size_t entry_point = 0;

  std::mt19937 rng_;

  float calculate_distance(const Vector &a, const Vector &b,
                           DistanceMetric metric) const;

  int random_level();

  std::size_t greedy_search_layer(const Vector &query, std::size_t entry,
                                  int level, DistanceMetric metric) const;

  std::vector<std::size_t> search_layer(const Vector &query,
                                        const std::vector<std::size_t> &entries,
                                        std::size_t ef, int level,
                                        DistanceMetric metric) const;

  std::vector<std::size_t>
  select_neighbours(const Vector &query,
                    const std::vector<std::size_t> &candidates,
                    std::size_t count, DistanceMetric metric) const;

public:
  explicit HnswIndex(std::size_t dimension, std::size_t M = 16,
                     std::size_t ef_construction = 100,
                     std::size_t ef_search = 50);

  void insert(std::size_t id, Vector vector);

  std::vector<SearchResult>
  search(const Vector &query, std::size_t k,
         DistanceMetric metric = DistanceMetric::SquaredL2) const;

  bool constains(std::size_t id) const;

  std::size_t size() const;
  std::size_t dimension() const;

  void set_ef_search(std::size_t ef_search);
};
}
