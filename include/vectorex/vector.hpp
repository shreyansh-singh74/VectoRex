#pragma once
#include <cstddef>
#include<vector>

namespace vectorex {
    class Vector{
        private:
            std::vector<float> data_;

        public:
        explicit Vector(std::vector<float> data);

        std::size_t size() const;

        float at(std::size_t index) const;

        const std::vector<float>& data() const;
    };
}
