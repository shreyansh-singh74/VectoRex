#pragma once

#include "vectorex/vector.hpp"
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace vectorex{

    struct SearchResult{
        std::size_t id;
        float distance;
    };

    enum class DistanceMetric{
        SquaredL2,
        Cosine,
        InnerProduct
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

            // void add(std::size_t id,Vector vector);

            std::vector<SearchResult> search(const Vector& query,std::size_t k, DistanceMetric metric = DistanceMetric::SquaredL2) const;

            std::size_t size() const;

            std::size_t dimension() const;

            void save(const std::string& path) const;

            static VectorStore load(const std::string& path);

            bool contains(std::uint64_t id) const;

            void remove(std::uint64_t id);

            void update(std::uint64_t id,const Vector& vector);
    };
}
