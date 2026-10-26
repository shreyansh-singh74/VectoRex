#include "vectorex/hnsw_index.hpp"
#include "vectorex/vector_store.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vectorex;

int main() {
    HnswIndex index(2, 4, 20, 10);

    index.insert(1, Vector({1.0f, 1.0f}));
    index.insert(2, Vector({2.0f, 2.0f}));
    index.insert(3, Vector({10.0f, 10.0f}));

    assert(index.size() == 3);
    assert(index.contains(1));

    auto results = index.search(Vector({1.1f, 1.1f}), 2);

    assert(results.size() == 2);
    assert(results[0].id == 1);

    bool duplicate_thrown = false;
    try {
        index.insert(1, Vector({3.0f, 3.0f}));
    } catch (const std::invalid_argument&) {
        duplicate_thrown = true;
    }
    assert(duplicate_thrown);

    bool dimension_thrown = false;
    try {
        index.insert(4, Vector({1.0f, 2.0f, 3.0f}));
    } catch (const std::invalid_argument&) {
        dimension_thrown = true;
    }
    assert(dimension_thrown);

    // Compare HNSW results against exact brute-force search.
    VectorStore exact(2);

    exact.insert(1, Vector({1.0f, 1.0f}));
    exact.insert(2, Vector({2.0f, 2.0f}));
    exact.insert(3, Vector({10.0f, 10.0f}));

    Vector query({1.1f, 1.1f});

    auto hnsw_results = index.search(query, 2);
    auto exact_results = exact.search(query, 2);

    assert(hnsw_results.size() == exact_results.size());

    for (std::size_t i = 0; i < exact_results.size(); ++i) {
        assert(hnsw_results[i].id == exact_results[i].id);
    }

    std::cout << "All HNSW tests passed.\n";
}
