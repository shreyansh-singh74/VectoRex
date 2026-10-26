#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "vectorex/vector.hpp"
#include "vectorex/vector_store.hpp"

void test_insert_one_vector() {
    vectorex::VectorStore store(3);
    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));
    auto r = store.search(vectorex::Vector({1.0, 2.0, 3.0}), 1);
    assert(r.size()==1 && r[0].id==101);
}
void test_insert_multiple_vectors() {
    vectorex::VectorStore store(3);
    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));
    store.insert(102, vectorex::Vector({4.0, 5.0, 6.0}));
    store.insert(103, vectorex::Vector({7.0, 8.0, 9.0}));
    auto r = store.search(vectorex::Vector({1.0, 2.0, 3.0}), 3);
    assert(r.size()==3);
}
void test_search_returns_closest_vector() {
    vectorex::VectorStore store(3);
    store.insert(101, vectorex::Vector({1.0, 1.0, 1.0}));
    store.insert(102, vectorex::Vector({5.0, 5.0, 5.0}));
    store.insert(103, vectorex::Vector({10.0,10.0,10.0}));
    auto r = store.search(vectorex::Vector({1.1,1.1,1.1}),1);
    assert(r[0].id==101);
}
void test_top_k_ordering() {
    vectorex::VectorStore store(2);
    store.insert(101, vectorex::Vector({1.0,1.0}));
    store.insert(102, vectorex::Vector({2.0,2.0}));
    store.insert(103, vectorex::Vector({5.0,5.0}));
    auto r = store.search(vectorex::Vector({1.0,1.0}),3);
    assert(r[0].id==101 && r[1].id==102 && r[2].id==103);
    assert(r[0].distance <= r[1].distance && r[1].distance <= r[2].distance);
}
void test_k_greater_than_number_of_vectors() {
    vectorex::VectorStore store(2);
    store.insert(101, vectorex::Vector({1.0,1.0}));
    store.insert(102, vectorex::Vector({2.0,2.0}));
    auto r = store.search(vectorex::Vector({1.0,1.0}),10);
    assert(r.size()==2);
}
void test_reject_wrong_dimension_on_insert() {
    vectorex::VectorStore store(3);
    bool threw=false; try{store.insert(101, vectorex::Vector({1.0,2.0}));}catch(const std::invalid_argument&){threw=true;}
    assert(threw);
}
void test_reject_wrong_dimension_on_search() {
    vectorex::VectorStore store(3);
    store.insert(101, vectorex::Vector({1.0,2.0,3.0}));
    bool threw=false; try{store.search(vectorex::Vector({1.0,2.0}),1);}catch(const std::invalid_argument&){threw=true;}
    assert(threw);
}
void test_ids_are_preserved() {
    vectorex::VectorStore store(2);
    store.insert(500, vectorex::Vector({1.0,1.0}));
    store.insert(900, vectorex::Vector({2.0,2.0}));
    auto r = store.search(vectorex::Vector({1.0,1.0}),2);
    assert(r[0].id==500 && r[1].id==900);
}
void test_empty_store() {
    vectorex::VectorStore store(3);
    auto r = store.search(vectorex::Vector({1.0,2.0,3.0}),5);
    assert(r.empty());
}
void test_identical_vector_distance_zero() {
    vectorex::VectorStore store(3);
    store.insert(101, vectorex::Vector({4.0,5.0,6.0}));
    auto r = store.search(vectorex::Vector({4.0,5.0,6.0}),1);
    assert(std::abs(r[0].distance) < 1e-9);
}

void test_truncated_file() {
    const std::string path = "truncated.dat";

    std::ofstream out(path, std::ios::binary);

    uint32_t magic = 0x56524558;
    uint32_t version = 1;
    uint64_t dimension = 3;
    uint64_t count = 1;

    out.write(
        reinterpret_cast<const char*>(&magic),
        sizeof(magic)
    );

    out.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version)
    );

    out.write(
        reinterpret_cast<const char*>(&dimension),
        sizeof(dimension)
    );

    out.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    // Write only the ID.
    // We deliberately DO NOT write the vector.
    uint64_t id = 101;

    out.write(
        reinterpret_cast<const char*>(&id),
        sizeof(id)
    );

    out.close();

    bool threw = false;

    try {
        vectorex::VectorStore::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::remove(path.c_str());
}

void test_invalid_magic() {
    const std::string path = "invalid_magic.dat";

    std::ofstream out(path, std::ios::binary);

    uint32_t bad_magic = 0x12345678;
    uint32_t version = 1;
    uint64_t dimension = 3;
    uint64_t count = 0;

    out.write(
        reinterpret_cast<const char*>(&bad_magic),
        sizeof(bad_magic)
    );

    out.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version)
    );

    out.write(
        reinterpret_cast<const char*>(&dimension),
        sizeof(dimension)
    );

    out.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    out.close();

    bool threw = false;

    try {
        vectorex::VectorStore::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::remove(path.c_str());
}

void test_unsupported_version() {
    const std::string path = "unsupported_version.dat";

    std::ofstream out(path, std::ios::binary);

    uint32_t magic = 0x56524558;
    uint32_t version = 999;
    uint64_t dimension = 3;
    uint64_t count = 0;

    out.write(
        reinterpret_cast<const char*>(&magic),
        sizeof(magic)
    );

    out.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version)
    );

    out.write(
        reinterpret_cast<const char*>(&dimension),
        sizeof(dimension)
    );

    out.write(
        reinterpret_cast<const char*>(&count),
        sizeof(count)
    );

    out.close();

    bool threw = false;

    try {
        vectorex::VectorStore::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    assert(threw);

    std::remove(path.c_str());
}

void test_large_id_persistence() {
    const std::string path = "large_id.dat";

    const uint64_t large_id = 5000000000ULL;

    vectorex::VectorStore original(3);

    original.insert(
        large_id,
        vectorex::Vector({1.0f, 2.0f, 3.0f})
    );

    original.save(path);

    vectorex::VectorStore loaded =
        vectorex::VectorStore::load(path);

    auto results = loaded.search(
        vectorex::Vector({1.0f, 2.0f, 3.0f}),
        1
    );

    assert(results.size() == 1);
    assert(results[0].id == large_id);

    std::remove(path.c_str());
}

void test_persistence() {
    const std::string path = "test_vector_store.dat";

    vectorex::VectorStore original(3);

    original.insert(101, vectorex::Vector({1.0f, 2.0f, 3.0f}));
    original.insert(103, vectorex::Vector({4.0f, 5.0f, 6.0f}));
    original.insert(105, vectorex::Vector({7.0f, 8.0f, 9.0f}));

    original.save(path);

    vectorex::VectorStore loaded =
        vectorex::VectorStore::load(path);

    assert(loaded.size() == original.size());
    assert(loaded.dimension() == original.dimension());

    auto r = loaded.search(
        vectorex::Vector({1.0f, 2.0f, 3.0f}),
        3
    );

    assert(r.size() == 3);
    assert(r[0].id == 101);
    assert(r[1].id == 103);
    assert(r[2].id == 105);

    // std::remove(path.c_str());
}

void test_duplicate_id_rejected() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    bool threw = false;

    try {
        store.insert(101, vectorex::Vector({4.0, 5.0, 6.0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
    assert(store.size() == 1);
}

void test_contains() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    assert(store.contains(101));
    assert(!store.contains(999));
}

void test_remove() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));
    store.insert(102, vectorex::Vector({4.0, 5.0, 6.0}));

    store.remove(101);

    assert(!store.contains(101));
    assert(store.contains(102));
    assert(store.size() == 1);
}

void test_update() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    store.update(101, vectorex::Vector({10.0, 20.0, 30.0}));

    auto result = store.search(
        vectorex::Vector({10.0, 20.0, 30.0}),
        1
    );

    assert(result.size() == 1);
    assert(result[0].id == 101);
    assert(result[0].distance == 0.0);
}


void test_remove_nonexistent_id() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    bool threw = false;

    try {
        store.remove(999);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
    assert(store.size() == 1);
}


void test_update_nonexistent_id() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    bool threw = false;

    try {
        store.update(999, vectorex::Vector({4.0, 5.0, 6.0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
    assert(store.size() == 1);
}

void test_update_dimension_mismatch() {
    vectorex::VectorStore store(3);

    store.insert(101, vectorex::Vector({1.0, 2.0, 3.0}));

    bool threw = false;

    try {
        store.update(101, vectorex::Vector({4.0, 5.0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
    assert(store.size() == 1);
}

void test_cosine_search() {
    vectorex::VectorStore store(2);

    store.insert(101, vectorex::Vector({1.0, 0.0}));
    store.insert(102, vectorex::Vector({0.0, 1.0}));
    store.insert(103, vectorex::Vector({0.9, 0.1}));

    auto result = store.search(
        vectorex::Vector({1.0, 0.0}),
        1,
        vectorex::DistanceMetric::Cosine
    );

    assert(result.size() == 1);
    assert(result[0].id == 101);
}

void test_inner_product_search() {
    vectorex::VectorStore store(2);

    store.insert(101, vectorex::Vector({1.0, 1.0}));
    store.insert(102, vectorex::Vector({5.0, 5.0}));
    store.insert(103, vectorex::Vector({2.0, 1.0}));

    auto result = store.search(
        vectorex::Vector({1.0, 1.0}),
        1,
        vectorex::DistanceMetric::InnerProduct
    );

    assert(result.size() == 1);
    assert(result[0].id == 102);
}


void test_hnsw_integration() {
    vectorex::VectorStore store(2);

    store.insert(1, vectorex::Vector({1.0f, 1.0f}));
    store.insert(2, vectorex::Vector({2.0f, 2.0f}));
    store.insert(3, vectorex::Vector({10.0f, 10.0f}));

    // Enable HNSW search.
    store.enable_hnsw();

    auto results = store.search(vectorex::Vector({1.1f, 1.1f}), 2);

    assert(results.size() == 2);
    assert(results[0].id == 1);
    assert(results[1].id == 2);

    // Updating a vector should refresh the HNSW index.
    store.update(1, vectorex::Vector({20.0f, 20.0f}));

    results = store.search(vectorex::Vector({1.1f, 1.1f}), 1);

    assert(results.size() == 1);
    assert(results[0].id == 2);

    // Removing a vector should refresh the HNSW index.
    store.remove(2);

    results = store.search(vectorex::Vector({1.1f, 1.1f}), 1);

    assert(results.size() == 1);
    assert(results[0].id == 3);

    // Disabling HNSW restores exact search.
    store.enable_hnsw(false);

    results = store.search(vectorex::Vector({1.1f, 1.1f}), 1);

    assert(results.size() == 1);
    assert(results[0].id == 3);
}




int main(){
    test_insert_one_vector(); test_insert_multiple_vectors(); test_search_returns_closest_vector();
    test_top_k_ordering(); test_k_greater_than_number_of_vectors(); test_reject_wrong_dimension_on_insert();
    test_reject_wrong_dimension_on_search(); test_ids_are_preserved(); test_empty_store(); test_identical_vector_distance_zero();
    test_persistence();
    test_invalid_magic();
    test_truncated_file();
    test_unsupported_version();
    test_large_id_persistence();
    test_duplicate_id_rejected();

    test_contains();
    test_remove();
    test_update();

    test_remove_nonexistent_id();
    test_update_nonexistent_id();
    test_update_dimension_mismatch();

    test_cosine_search();
    test_inner_product_search();

    test_hnsw_integration();


    std::cout<<"All VectorStore tests passed!\n";
}
