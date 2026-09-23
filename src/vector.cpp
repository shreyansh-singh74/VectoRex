#include "vectorex/vector.hpp"
#include<utility>
#include <cstddef>

// constructor
Vector::Vector(std::vector<float> data):data_(std::move(data)){

}

// size function
std::size_t Vector::size() const {
    return data_.size();
}

// access element
float Vector::at(std::size_t index) const {
    return data_[index];
}

//get underlying vector
const std::vector<float>& Vector::data() const{
    return data_;
}
