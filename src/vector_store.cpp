#include "vectorex/vector_store.hpp"
#include "vectorex/distance.hpp"
#include "vectorex/vector.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vectorex{

    void VectorStore::add(std::size_t id,Vector vector){

        if(!entries_.empty() && vector.size() != entries_.front().vector.size()){
            throw std::invalid_argument("Cannot add vector with different dimension");
        }

        entries_.push_back({
            id,
            std::move(vector)
        });
    }

    std::vector<SearchResult> VectorStore::search(const Vector& query,std::size_t k) const{

        if(!entries_.empty() && query.size() != entries_.front().vector.size()){
            throw std::invalid_argument("Cannot add vector with different dimension");
        }

        if(k==0 || entries_.empty()){
            return {};
        }

        std::vector<SearchResult> results;

        results.reserve(entries_.size());

        for(const auto& entry : entries_){
            float distance = squared_l2_distance(query, entry.vector);

            results.push_back({entry.id,distance});
        }

        std::sort(results.begin(),results.end(),[](const SearchResult& a,const SearchResult& b){
            return a.distance<b.distance;
        });

        if(k<results.size()){
            results.resize(k);
        }

        return results;
    }

    std::size_t VectorStore::size() const{
        return entries_.size();
    }

}
