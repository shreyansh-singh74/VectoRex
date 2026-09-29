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

    original.add(
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

    original.add(101, vectorex::Vector({1.0f, 2.0f, 3.0f}));
    original.add(103, vectorex::Vector({4.0f, 5.0f, 6.0f}));
    original.add(105, vectorex::Vector({7.0f, 8.0f, 9.0f}));

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


int main(){
    test_insert_one_vector(); test_insert_multiple_vectors(); test_search_returns_closest_vector();
    test_top_k_ordering(); test_k_greater_than_number_of_vectors(); test_reject_wrong_dimension_on_insert();
    test_reject_wrong_dimension_on_search(); test_ids_are_preserved(); test_empty_store(); test_identical_vector_distance_zero();
    test_persistence();
    test_invalid_magic();
    test_truncated_file();
    test_unsupported_version();
    test_large_id_persistence();

    std::cout<<"All VectorStore tests passed!\n";
}
