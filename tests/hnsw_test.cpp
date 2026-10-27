#include "vectorex/hnsw_index.hpp"
#include "vectorex/vector_store.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vectorex;

void test_hnsw_persistence() {
    const std::string path = "hnsw_persistence_test.dat";

    // Create and populate a store.
    VectorStore store(2);
    store.insert(101, Vector({1.0f, 1.0f}));
    store.insert(102, Vector({5.0f, 5.0f}));
    store.insert(103, Vector({9.0f, 9.0f}));

    store.enable_hnsw();

    // Save the vector store.
    store.save(path);

    // Load the saved vectors.
    VectorStore loaded = VectorStore::load(path);

    assert(loaded.size() == 3);
    assert(loaded.dimension() == 2);
    assert(loaded.contains(101));
    assert(loaded.contains(102));
    assert(loaded.contains(103));

    // Rebuild the HNSW index after loading.
    loaded.enable_hnsw();

    // Verify nearest-neighbor search.
    auto results = loaded.search(Vector({1.1f, 1.1f}), 2);

    assert(results.size() == 2);
    assert(results[0].id == 101);

    // Clean up the test file.
    std::remove(path.c_str());
}

void test_vector_store_hnsw_integration() {
    VectorStore store(2);

    store.insert(10, Vector({1.0f, 1.0f}));
    store.insert(20, Vector({5.0f, 5.0f}));
    store.insert(30, Vector({9.0f, 9.0f}));

    // Enable HNSW after inserting vectors.
    store.enable_hnsw();

    auto results = store.search(Vector({1.1f, 1.1f}), 1);
    assert(results.size() == 1);
    assert(results[0].id == 10);

    // Update a vector and verify the index reflects the change.
    store.update(10, Vector({8.0f, 8.0f}));

    results = store.search(Vector({8.1f, 8.1f}), 1);
    assert(results.size() == 1);
    assert(results[0].id == 10);

    // Remove a vector and verify it no longer appears.
    store.remove(10);

    assert(!store.contains(10));
    assert(store.size() == 2);

    results = store.search(Vector({8.1f, 8.1f}), 2);
    assert(results.size() == 2);

    for (const auto& result : results) {
        assert(result.id != 10);
    }

    // Disable HNSW and verify exact search still works.
    store.enable_hnsw(false);

    results = store.search(Vector({8.1f, 8.1f}), 2);
    assert(results.size() == 2);
    assert(results[0].id == 30);
}

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

    test_vector_store_hnsw_integration();
    test_hnsw_persistence();



    std::cout << "All HNSW tests passed.\n";
}
