#include <cassert>
#include <iostream>

#include "vectorex/vector.hpp"

int main() {
    vectorex::Vector vec({10.0f, 20.0f, 30.0f});

    assert(vec.size() == 3);

    assert(vec.at(0) == 10.0f);
    assert(vec.at(1) == 20.0f);
    assert(vec.at(2) == 30.0f);

    const auto& data = vec.data();

    assert(data.size() == 3);
    assert(data[0] == 10.0f);
    assert(data[1] == 20.0f);
    assert(data[2] == 30.0f);

    std::cout << "Vector tests passed!\n";

    return 0;
}
