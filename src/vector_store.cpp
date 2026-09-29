#include "vectorex/vector_store.hpp"
#include "vectorex/distance.hpp"
#include "vectorex/vector.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <vector>

namespace vectorex{

    VectorStore::VectorStore(std::size_t dimension): dimension_(dimension){

    }

    void VectorStore::insert(std::size_t id,Vector vector){
        if(vector.size() != dimension_){
            throw std::invalid_argument("Dimension mismatch on insert");
        }
        entries_.push_back({id,std::move(vector)});
    }

    void VectorStore::add(std::size_t id,Vector vector){
        insert(id, std::move(vector));
    }

    std::vector<SearchResult> VectorStore::search(const Vector& query,std::size_t k) const{
        if(query.size() != dimension_){
            throw std::invalid_argument("Dimension mismatch on search");
        }
        if(k == 0 || entries_.empty()) return {};

        // fast if -> n <= k
        if(k>= entries_.size()){
            std::vector<SearchResult> r;
            r.reserve(entries_.size());

            for(auto& e : entries_){
                r.push_back({e.id,squared_l2_distance(query, e.vector)});
            }

            std::sort(r.begin(),r.end(),[](auto& a, auto& b){
                return a.distance < b.distance;
            });

            return r;
        }

        // using heap for -> large n > k
        auto cmp = [](const SearchResult& a,const SearchResult& b){
            return a.distance < b.distance;
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>,decltype(cmp)> heap(cmp);

        for(auto& e : entries_){
            SearchResult cur{e.id,squared_l2_distance(query, e.vector)};
            if(heap.size()<k){
                heap.push(cur);
            }
            else if(cur.distance<heap.top().distance){
                heap.pop();
                heap.push(cur);
            }
        }

        std::vector<SearchResult> res;
        res.reserve(heap.size());
        while(!heap.empty()){
            res.push_back(heap.top());
            heap.pop();
        }
        std::sort(res.begin(),res.end(),[](auto& a,auto& b){
            return a.distance < b.distance;
        });

        return res;
    }

    std::size_t VectorStore::size() const{
        return entries_.size();
    }

    std::size_t VectorStore::dimension() const{
        return dimension_;
    }

    void VectorStore::save(const std::string& path) const{
        std::ofstream out(path,std::ios::binary);
        if(!out) throw std::runtime_error("Cannot open file for save");
        const uint32_t magic = 0x56524558;
        const uint32_t version = 1;
        uint64_t dim = dimension_;
        uint64_t count = entries_.size();

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for(auto& e:entries_){
            uint32_t id = e.id;
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out.write(reinterpret_cast<const char*>(e.vector.data().data()), dim*sizeof(float));
        }
        if(!out){
            throw std::runtime_error("Failed to write file");
        }
    }

    VectorStore VectorStore::load(const std::string &path){
        std::ifstream in(path,std::ios::binary);
        if(!in){
            throw std::runtime_error("Cannot open file for load");
        }
    }
}
