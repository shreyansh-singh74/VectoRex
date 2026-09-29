#include <cassert>
#include <cmath>
#include <iostream>

#include "vectorex/distance.hpp"

bool nearly_equal(float a, float b, float epsilon = 0.0001f) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    vectorex::Vector a({1.0f, 2.0f, 3.0f});
    vectorex::Vector b({4.0f, 6.0f, 3.0f});

    assert(nearly_equal(
        vectorex::squared_l2_distance(a, b),
        25.0f
    ));

    assert(nearly_equal(
        vectorex::l2_distance(a, b),
        5.0f
    ));

    assert(nearly_equal(
        vectorex::inner_product(a, b),
        25.0f
    ));

    assert(nearly_equal(
        vectorex::cosine_similarity(a, b),
        0.855482f
    ));

    std::cout << "Distance tests passed!\n";

    return 0;
}
