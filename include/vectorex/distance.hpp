#pragma once
#include "vectorex/vector.hpp"

namespace vectorex{
    float l2_distance(const Vector& a,const Vector& b);

    float squared_l2_distance(const Vector& a,const Vector& b);

    float cosine_similarity(const Vector& a,const Vector& b);

    float inner_product(const Vector& a,const Vector& b);
}
