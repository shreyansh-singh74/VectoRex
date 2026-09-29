#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./VectoRexInspector <file>\n";
        return 1;
    }

    const char* path = argv[1];

    std::ifstream in(path, std::ios::binary);

    if (!in) {
        std::cerr << "Cannot open file: " << path << "\n";
        return 1;
    }

    uint32_t magic;
    uint32_t version;
    uint64_t dimension;
    uint64_t count;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (!in) {
        std::cerr << "Corrupted file: incomplete header\n";
        return 1;
    }

    std::cout << "VectoRex Database\n";
    std::cout << "=================\n";

    std::cout << "Magic:     0x"
              << std::hex << magic << std::dec << "\n";

    std::cout << "Version:   " << version << "\n";
    std::cout << "Dimension: " << dimension << "\n";
    std::cout << "Vectors:   " << count << "\n\n";

    std::cout << "Records\n";
    std::cout << "-------\n";

    for (uint64_t i = 0; i < count; ++i) {
        uint64_t id;

        in.read(reinterpret_cast<char*>(&id), sizeof(id));

        if (!in) {
            std::cerr << "Corrupted file: incomplete ID\n";
            return 1;
        }

        std::vector<float> data(dimension);

        in.read(
            reinterpret_cast<char*>(data.data()),
            dimension * sizeof(float)
        );

        if (!in) {
            std::cerr << "Corrupted file: incomplete vector\n";
            return 1;
        }

        std::cout << "ID: " << id << "\n";
        std::cout << "Vector: [";

        for (uint64_t j = 0; j < dimension; ++j) {
            std::cout << data[j];

            if (j + 1 < dimension) {
                std::cout << ", ";
            }
        }

        std::cout << "]\n\n";
    }

    return 0;
}
