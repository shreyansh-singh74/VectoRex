#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "vectorex/vector_store.hpp"

using namespace vectorex;

Vector parse_vector(std::istringstream& input){
    std::vector<float> values;
    float value;

    while(input >> value){
        values.push_back(value);
    }

    if(values.empty()){
        throw std::invalid_argument("Provide at least one vector value.");
    }
    return Vector(std::move(values));
}

void print_help(){
    std::cout
        << "\nVectoRex CLI Commands\n"
        << "  insert <id> <v1> <v2> ...  Insert a vector\n"
        << "  search <k> <v1> <v2> ...  Search nearest vectors\n"
        << "  update <id> <v1> <v2> ... Update a vector\n"
        << "  delete <id>               Delete a vector\n"
        << "  contains <id>             Check whether an ID exists\n"
        << "  save <file>               Save the store\n"
        << "  load <file>               Load a store from disk\n"
        << "  hnsw on|off               Enable or disable HNSW search\n"
        << "  ef <value>                Set HNSW search breadth\n"
        << "  help                      Show this help\n"
        << "  exit                      Quit\n\n";
}

int main() {
    std::unique_ptr<VectorStore> store;
    std::string line;

    std::cout<<"VectoRex Interactive CLI\n";
    std::cout<<"Create a store with: creaet <dimension>\n";
    std::cout<<"Type 'help' for commands.\n";

    while(true){
        std::cout<<"vectorex> ";
        if(!std::getline(std::cin, line)){
            break;
        }

        std::istringstream input(line);
        std::string command;

        if(!(input>>command)){
            continue;
        }

        try{
            if(command == "exit" || command == "quit"){
                break;
            }

            if(command == "help"){
                print_help();
                std::cout<<"  create <dimension>          Create a new empty store\n";
                continue;
            }

            if(command == "create"){
                std::size_t dimension;

                if(!(input>>dimension) || dimension==0){
                    std::cout<<"Usage: create <positive_dimension>\n";
                    continue;
                }

                store = std::make_unique<VectorStore>(dimension);
                std::cout<<"Created vector store with dimension>\n"<<dimension<<".\n";
                continue;
            }

            if(command == "load"){
                std::string path;
                if(!(input>>path)){
                    std::cout<<"Usage: load <file>\n";
                    continue;
                }
                store = std::make_unique<VectorStore>(VectorStore::load(path));

                std::cout<<"Loaded "<<store->size()<<" vectors, dimension "<<store->dimension()<<".\n";
                continue;
            }

            if(!store){
                std::cout<<"Create or load a store first.\n";
                continue;
            }

            if(command=="insert"){
                std::size_t id;
                if(!(input>>id)){
                    std::cout << "Usage: insert <id> <v1> <v2> ...\n";
                    continue;
                }

                Vector vector = parse_vector(input);
                store->insert(id, std::move(vector));
                std::cout<<" Inserted vector "<<id<<".\n";
            }
            else if(command == "search"){
                std::size_t k;
                if(!(input>>k)){
                    std::cout<<"Usage: search <k> <v1> <v2> ... \n";
                    continue;
                }

                Vector query = parse_vector(input);
                auto results = store->search(query, k);

                if(results.empty()){
                    std::cout<<"No results found.\n";
                }

                for(const auto& result : results){
                    std::cout<<"ID: "<<result.id<<", Distance: "<<result.distance<<"\n";
                }
            }
            else if(command == "update"){
                std::size_t id;
                if(!(input>>id)){
                    std::cout << "Usage: update <id> <v1> <v2> ...\n";
                    continue;
                }

                Vector vector = parse_vector(input);
                store->update(id, vector);
                std::cout<<"Updated vector "<<id<<".\n";
            }
            else if(command=="delete"){
                std::size_t id;
                if(!(input>>id)){
                    std::cout << "Usage: delete <id>\n";
                    continue;
                }

                store->remove(id);
                std::cout << "Removed vector " << id << ".\n";
            }
            else if(command=="contains"){
                std::size_t id;
                if(!(input>>id)){
                    std::cout<<"Usage: contains <id>\n";
                    continue;
                }

                std::cout<<(store->contains(id)?"Found\n":"Not Found\n");
            }
            else if(command=="save"){
                std::string path;
                if(!(input>>path)){
                    std::cout<<"Usage: save <file>\n";
                    continue;
                }

                store->save(path);
            }
            else if(command=="hnsw"){
                std::string state;
                if(!(input>>state) || (state!= "on" && state!="off")){
                    std::cout << "Usage: hnsw on|off\n";
                    continue;
                }

                store->enable_hnsw(state=="on");
                std::cout<<"HNSW"<<state<<"./n";
            }
            else if(command=="ef"){
                std::size_t value;
                if(!(input>>value) || value==0){
                    std::cout << "Usage: ef <positive_value>\n";
                    continue;
                }

                store->set_hnsw_ef_search(value);
                std::cout<<"HNSW ef_search set to "<<value<<".\n";
            }
            else{
                std::cout<<"Unknown command. Type 'help'.\n";
            }
        }
        catch(const std::exception& error){
            std::cout<<"Error: "<<error.what()<<"\n";
        }
    }

    std::cout<<"Exiting VectoRex.\n";

    return 0;
}
