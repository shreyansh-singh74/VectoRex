#include "vectorex/vector_store.hpp"
#include "vectorex/vector.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace vectorex;
using Clock = std::chrono::high_resolution_clock;

Vector random_vector(std::size_t dim, std::mt19937& rng){
    std::uniform_real_distribution<float> d(-1.0f,1.0f);
    std::vector<float> v; v.reserve(dim);
    for(std::size_t i=0;i<dim;++i) v.push_back(d(rng));
    return Vector(std::move(v));
}

void run(std::size_t N, std::size_t dim, std::size_t k, std::size_t nq=100){
    std::mt19937 rng(42);
    VectorStore store(dim);
    auto t0 = Clock::now();
    for(std::size_t i=0;i<N;++i) store.insert(i, random_vector(dim,rng));
    auto t1 = Clock::now();
    double insert_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    std::vector<Vector> queries; for(std::size_t i=0;i<nq;++i) queries.push_back(random_vector(dim,rng));
    t0 = Clock::now();
    for(auto& q: queries) store.search(q,k);
    auto t2 = Clock::now();
    double search_ms = std::chrono::duration<double,std::milli>(t2-t0).count();
    double qps = nq / (search_ms/1000.0);
    double mem_mb = (double)N*dim*4 / (1024*1024);

    std::cout << "N=" << N << " dim=" << dim << " k=" << k
              << " | insert " << insert_ms << " ms"
              << " | search " << nq << " queries " << search_ms << " ms"
              << " avg " << search_ms/nq << " ms/q"
              << " QPS " << qps
              << " | mem ~" << mem_mb << " MB\n";
}
int main(){
    std::cout << "BruteForce Benchmark - heap O(N log K)\n";
    for(auto N: {1000,10000,100000}) run(N, 64, 10);
    // 1M is heavy - uncomment if needed: run(1000000, 64, 10);
}
