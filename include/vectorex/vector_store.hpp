#pragma once

#include "vectorex/vector.hpp"
#include <cstddef>
#include <vector>

namespace vectorex{

    struct SearchResult{
        std::size_t id;
        float distance;
    };

    class VectorStore{
        private:
            struct Entry{
                std::size_t id;
                Vector vector;
            };

            std::vector<Entry> entries_;

        public:
            void add(std::size_t id,Vector vector);

            std::vector<SearchResult> search(const Vector& query,std::size_t k) const;

            std::size_t size() const;
    };

}
