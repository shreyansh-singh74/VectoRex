#include "vectorex/types.hpp"
#include "vectorex/vector.hpp"
#include "vectorex/vector_store.hpp"
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace vectorex;
using Clock = std::chrono::high_resolution_clock;

Vector random_vector(std::size_t dim, std::mt19937 &rng) {
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> values;
  values.reserve(dim);

  for (std::size_t i = 0; i < dim; ++i) {
    values.push_back(distribution(rng));
  }
  return Vector(std::move(values));
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void run(std::size_t N, std::size_t dim, std::size_t k, std::size_t nq = 100) {
  std::mt19937 rng(42);

  std::vector<Vector> dataset;
  dataset.reserve(N);

  for (std::size_t i = 0; i < N; i++) {
    dataset.push_back(random_vector(dim, rng));
  }

  std::vector<Vector> queries;
  queries.reserve(nq);

  for (std::size_t i = 0; i < nq; i++) {
    queries.push_back(random_vector(dim, rng));
  }

  VectorStore store(dim);

  auto start = Clock::now();

  for (std::size_t i = 0; i < N; i++) {
    store.insert(i, dataset[i]);
  }

  auto end = Clock::now();
  double insert_ms = elapsed_ms(start, end);

  std::vector<std::vector<SearchResult>> exact_results;
  exact_results.reserve(nq);

  start = Clock::now();

  for (const auto &query : queries) {
    exact_results.push_back(store.search(query, k));
  }

  end = Clock::now();

  double exact_search_ms = elapsed_ms(start, end);

  start = Clock::now();
  store.enable_hnsw(true);
  store.set_hnsw_ef_search(200);
  end = Clock::now();

  double hnsw_build_ms = elapsed_ms(start, end);

  std::size_t matching_results = 0;
  std::size_t total_exact_results = 0;

  start = Clock::now();
  for (std::size_t i = 0; i < nq; i++) {
    auto approximate_results = store.search(queries[i], k);

    std::unordered_set<std::size_t> exact_ids;
    for (const auto &result : exact_results[i]) {
      exact_ids.insert(result.id);
    }

    for (const auto &result : approximate_results) {
      if (exact_ids.count(result.id) > 0) {
        matching_results++;
      }
    }

    total_exact_results += exact_results[i].size();
  }

  end = Clock::now();

  double hnsw_search_ms = elapsed_ms(start, end);

  double recall = total_exact_results == 0
                      ? 0.0
                      : 100.0 * matching_results / total_exact_results;

  double exact_avg_ms = exact_search_ms / nq;
  double hnsw_avg_ms = hnsw_search_ms / nq;

  double exact_qps = exact_search_ms > 0 ? static_cast<double>(nq) * 1000.0 / exact_search_ms : 0.0;

  double hnsw_qps = hnsw_search_ms > 0 ? static_cast<double>(nq) * 1000.0 / hnsw_search_ms : 0.0;

  double speedup = hnsw_avg_ms > 0 ? exact_avg_ms / hnsw_avg_ms : 0.0;

  double data_mb = static_cast<double>(N)*dim*sizeof(float)/(1024.0*1024.0);

  std::cout<<"\n=============================\n"
           <<"Dataset: N="<<N
           <<", dimensions="<<dim
           <<", K="<<k
           <<", queries="<<nq<<"\n";

  std::cout<<std::fixed<<std::setprecision(3);

  std::cout<<"\nIndexing:\n";
  std::cout<<" Vector insertion: "<<insert_ms<<" ms\n";
  std::cout<<" HNSW graph buld: "<<hnsw_build_ms<<" ms\n";

  std::cout<<"\nExact brute force search:\n";
  std::cout<<" Total: "<<exact_search_ms<<" ms\n";
  std::cout<<" Average: "<<exact_avg_ms<<" ms/query\n";
  std::cout<<" QPS: "<<exact_qps<<"\n";

  std::cout<<"\nHNSW search :\n";
  std::cout<<" Total: "<<hnsw_search_ms<<" ms\n";
  std::cout<<" Average: "<<hnsw_avg_ms<<" ms/query\n";
  std::cout<<" QPS: "<<hnsw_qps<<"\n";

  std::cout<<"\nQuality and memory:\n";
  std::cout<<" Recall@K: "<<recall<<"%\n";
  std::cout<<" Search speedup: "<<speedup<<"x\n";
  std::cout<<" Vector data size: "<<data_mb<<" MB\n";

}

int main() {
    std::cout<<"VectoRex: Brute Force vs HNSW Benchmark\n";

    run(1000,64,10);
    run(10000,64,10);
    run(100000,64,10);
}
