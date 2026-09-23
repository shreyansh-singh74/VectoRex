#include <iostream>

#include "vectorex/vector_store.hpp"

int main() {
    vectorex::VectorStore store;

    store.add(101, Vector({1.0f, 2.0f, 3.0f}));
    store.add(102, Vector({4.0f, 6.0f, 3.0f}));
    store.add(103, Vector({2.0f, 2.0f, 4.0f}));

    Vector query({1.0f, 2.0f, 3.0f});

    auto results = store.search(query, 2);

    for (const auto& result : results) {
        std::cout
            << "ID: " << result.id
            << ", Distance: " << result.distance
            << '\n';
    }

    return 0;
}
