#pragma once

#include <array>
#include <cassert>
#include <numeric>
#include <optional>
#include <vector>

namespace haki {
    constexpr size_t kBlockSize = 16;

    class BlockAllocator {
    public:
        BlockAllocator(size_t max_blocks)
            : max_blocks(max_blocks)
            , free_blocks(max_blocks)
            , in_use(max_blocks, 0) {
            std::ranges::iota(free_blocks.begin(), free_blocks.end(), 0UZ);
        }

        [[nodiscard]] std::optional<size_t> allocate() {
            if (free_blocks.size() == 0) {
                return std::nullopt;
            }
            size_t id = free_blocks.back();
            in_use[id] = 1;

            free_blocks.pop_back();
            return id;
        }

        void free(size_t id) {
            assert(id < max_blocks);
            assert(in_use[id]);

            in_use[id] = 0;
            free_blocks.push_back(id);
        }

        [[nodiscard]] size_t num_free() const { return free_blocks.size(); }

    private:
        size_t max_blocks;
        std::vector<size_t> free_blocks;
        std::vector<uint8_t> in_use;
    };
}
