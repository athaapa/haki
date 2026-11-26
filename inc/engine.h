#pragma once

#include "block_allocator.h"
#include <array>
#include <cassert>
#include <deque>
#include <expected>
#include <numeric>
#include <span>
#include <vector>

namespace haki {
    struct Sequence {
        std::span<const int> tokens;
        std::vector<size_t> physical_blocks;
        size_t num_computed_tokens;
    };

    enum class ScheduleError : uint8_t { OutOfSequences, QueueEmpty };

    constexpr size_t kMaxSequences = 16;
    constexpr size_t kMaxBatchedTokens = 512;

    struct SchedulerOutput {
        std::vector<size_t> seq_ids;
        std::vector<size_t> num_scheduled;
        std::vector<size_t> context_lens;

        std::vector<int32_t> slot_mapping;
        std::vector<int32_t> positions;

        std::vector<std::span<const size_t>> block_tables;
    };

    template <std::floating_point T> class Engine {
    public:
        Engine()
            : free_list(kMaxSequences) {
            running.reserve(kMaxSequences);
            std::ranges::iota(free_list.begin(), free_list.end(), 0);
        }

        [[nodiscard]] std::expected<void, ScheduleError> submit(std::span<const int> tokens) {
            if (free_list.empty()) {
                return std::unexpected(ScheduleError::OutOfSequences);
            }

            size_t id = free_list.back();
            free_list.pop_back();

            sequences[id].tokens = tokens;
            sequences[id].physical_blocks.clear();
            sequences[id].num_computed_tokens = 0;

            waiting.push_back(id);

            return { };
        }

        /*
         * Schedule decodes and prefills
         */
        SchedulerOutput schedule() {
            SchedulerOutput out;

            size_t remaining_budget = kMaxBatchedTokens;

            bool preempted = false;
            auto schedule_sequence
                = [&out, &remaining_budget, this](size_t seq_id, size_t scheduled_tokens) {
                      auto& seq = sequences[seq_id];
                      remaining_budget -= scheduled_tokens;

                      size_t blocks_needed = compute_blocks_needed(seq, scheduled_tokens);

                      allocate_blocks(
                          seq, blocks_needed); // we did not preempt ourselves, so we can allocate

                      out.seq_ids.push_back(seq_id);
                      out.block_tables.emplace_back(seq.physical_blocks);
                      out.num_scheduled.push_back(scheduled_tokens);

                      for (size_t t = 0; t < scheduled_tokens; ++t) {
                          size_t pos = seq.num_computed_tokens + t;
                          size_t phys_block = seq.physical_blocks[pos / kBlockSize];
                          size_t offset = pos % kBlockSize;
                          size_t slot = (phys_block * kBlockSize) + offset;

                          out.slot_mapping.push_back(static_cast<int32_t>(slot));
                          out.positions.push_back(static_cast<int32_t>(pos));
                      }

                      seq.num_computed_tokens += scheduled_tokens;
                      out.context_lens.push_back(seq.num_computed_tokens);
                  };

            for (size_t i = 0; i < running.size(); ++i) {
                size_t seq_id = running[i];
                auto& seq = sequences[seq_id];
                size_t num_uncomputed_tokens = seq.tokens.size() - seq.num_computed_tokens;
                size_t scheduled_tokens = std::min(num_uncomputed_tokens, remaining_budget);

                size_t blocks_needed = compute_blocks_needed(seq, scheduled_tokens);

                while (block_allocator.num_free() < blocks_needed) {
                    if (running.size() == 1) {
                        running.pop_back();
                        return { };
                    }
                    preempt_victim();
                    preempted = true;
                    if (i >= running.size()) {
                        break;
                    }
                }

                if (i < running.size()) {
                    schedule_sequence(seq_id, scheduled_tokens);
                }

                if (remaining_budget == 0) {
                    break;
                }
            }

            if (!preempted) {
                while (!waiting.empty()) {
                    size_t seq_id = waiting.front();
                    auto& seq = sequences[seq_id];
                    size_t num_uncomputed_tokens = seq.tokens.size() - seq.num_computed_tokens;
                    size_t scheduled_tokens = std::min(num_uncomputed_tokens, remaining_budget);

                    size_t blocks_needed = compute_blocks_needed(seq, scheduled_tokens);

                    if (block_allocator.num_free()
                        < blocks_needed) { // dont evict running sequences to schedule a waiting one
                        break;
                    }

                    schedule_sequence(seq_id, scheduled_tokens);

                    waiting.pop_front();
                    running.push_back(seq_id);

                    if (remaining_budget == 0) {
                        break;
                    }
                }
            }

            return out;
        }

    private:
        std::array<Sequence, kMaxSequences> sequences;
        std::vector<size_t> free_list;

        std::deque<size_t> waiting;
        std::vector<size_t> running;

        BlockAllocator block_allocator;

        void preempt_victim() {
            size_t seq_id = running.back();
            auto& seq = sequences[seq_id];

            free_all_blocks(seq);
            seq.num_computed_tokens = 0;

            running.pop_back();
            waiting.push_front(seq_id);
        }

        void allocate_blocks(Sequence& seq, size_t k) {
            for (size_t b = 0; b < k; ++b) {
                seq.physical_blocks.push_back(*block_allocator.allocate());
            }
        }

        void free_all_blocks(Sequence& seq) {
            for (size_t b : seq.physical_blocks) {
                block_allocator.free(b);
            }
            seq.physical_blocks.clear();
        }

        void complete_sequence(size_t running_idx) {
            size_t seq_id = running[running_idx];
            auto& seq = sequences[seq_id];

            free_all_blocks(seq);

            running[running_idx] = running.back();
            running.pop_back();

            free_list.push_back(seq_id);
        }

        static size_t compute_blocks_needed(Sequence& seq, size_t scheduled_tokens) {
            size_t num_uncomputed_tokens = seq.tokens.size() - seq.num_computed_tokens;

            size_t num_slots = seq.physical_blocks.size() * kBlockSize;
            size_t slots_needed = seq.num_computed_tokens + scheduled_tokens;

            size_t blocks_needed = (slots_needed > num_slots)
                ? (slots_needed - num_slots + kBlockSize - 1) / kBlockSize
                : 0;

            return blocks_needed;
        }

        /*
         * Schedule a single-token decode for sequence i
         */
    };
}
