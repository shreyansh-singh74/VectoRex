#include "vectorex/distance.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace vectorex {
    float squared_l2_distance(const Vector &a, const Vector &b){
        if(a.size() != b.size()){
            throw std::invalid_argument("Cannot calculate distance between vectors of different dimensions");
        }

        float sum = 0.0f;

        for(std::size_t i=0;i<a.size();i++){
            float diff = a.at(i) - b.at(i);
            sum += diff*diff;
        }

        return sum;
    }

    float l2_distance(const Vector &a, const Vector &b){
        return std::sqrt((squared_l2_distance(a,b)));
    }

    float inner_product(const Vector &a, const Vector &b){
        if(a.size() != b.size()){
            throw std::invalid_argument("Cannot calculate distance between vectors of different dimensions");
        }

        float result = 0.0f;

        for(std::size_t i=0;i<a.size();i++){
            result += a.at(i) * b.at(i);
        }

        return result;
    }

    float cosine_similarity(const Vector &a, const Vector &b){
        if(a.size() != b.size()){
            throw std::invalid_argument("Cannot calculate distance between vectors of different dimensions");
        }

        float dot = 0.0f;
        float magnitude_a = 0.0f;
        float magnitude_b = 0.0f;

        for(std::size_t i=0;i<a.size();i++){
            float value_a  = a.at(i);
            float value_b = b.at(i);

            dot += value_a*value_b;
            magnitude_a += value_a*value_a;
            magnitude_b += value_b*value_b;
        }

        if(magnitude_a == 0.0f || magnitude_b == 0.0f){
            throw std::invalid_argument("Cosine Similarity is undefiend for a zero vector");
        }

        return dot / (std::sqrt(magnitude_a)*std::sqrt(magnitude_b));
    }
}
