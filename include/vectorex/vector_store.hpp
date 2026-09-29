#pragma once

#include "vectorex/vector.hpp"
#include <cstddef>
#include <iterator>
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
            std::size_t dimension_;

        public:

            explicit VectorStore(std::size_t dimension);

            void insert(std::size_t id, Vector vector);

            void add(std::size_t id,Vector vector);

            std::vector<SearchResult> search(const Vector& query,std::size_t k) const;

            std::size_t size() const;

            std::size_t dimension() const;

            void save(const std::string& path) const;

            static VectorStore load(const std::string& path);
    };
}
