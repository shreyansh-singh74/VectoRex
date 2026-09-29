#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
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
int main(){
    test_insert_one_vector(); test_insert_multiple_vectors(); test_search_returns_closest_vector();
    test_top_k_ordering(); test_k_greater_than_number_of_vectors(); test_reject_wrong_dimension_on_insert();
    test_reject_wrong_dimension_on_search(); test_ids_are_preserved(); test_empty_store(); test_identical_vector_distance_zero();
    std::cout<<"All VectorStore tests passed!\n";
}
