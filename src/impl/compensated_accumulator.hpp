#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "compensated_arithmetic.hpp"

namespace compensated_blas {

enum class accumulator_layout { soa, aos };

// Accumulator with configurable compensation bins per entry.
// Acts as a non-owning view over caller-managed storage.
template <typename T, accumulator_layout Layout = accumulator_layout::soa>
class compensated_accumulator_t {
public:
    compensated_accumulator_t() = default;

    compensated_accumulator_t(T *compensation_block,
                              std::size_t length,
                              std::size_t compensation_terms)
        : size_(length),
          compensation_terms_(compensation_terms),
          compensation_block_(compensation_block) {}

    void reset(T *compensation_block,
               std::size_t length,
               std::size_t compensation_terms) {
        size_ = length;
        compensation_terms_ = compensation_terms;
        compensation_block_ = compensation_block;
    }

    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t compensation_terms() const noexcept { return compensation_terms_; }

    inline T compensation(std::size_t bin, std::size_t i) const {
        assert(bin < compensation_terms_);
        assert(i < size_);
        assert(compensation_block_ != nullptr);
        return compensation_block_[compensation_offset(bin, i)];
    }

    inline void accumulate(std::size_t i, T &primary, const T &val) {
        assert(i < size_);
        if (compensation_terms_ == 0 || compensation_block_ == nullptr) {
            primary += val;
            return;
        }

        T carry = val;
        two_sum(primary, carry);
        for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
            if (carry == T(0)) {
                return;
            }
            two_sum(compensation_block_[compensation_offset(bin, i)], carry);
        }
        if (carry != T(0)) {
            primary += carry;
        }
    }

    inline T round(std::size_t i, T &primary) {
        assert(i < size_);
        if (compensation_terms_ == 0 || compensation_block_ == nullptr) {
            return primary;
        }

        T sum = primary;
        T carry = T(0);
        for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
            const std::size_t offset = compensation_offset(bin, i);
            const T value = compensation_block_[offset] + carry;
            compensation_block_[offset] = T(0);
            carry = value;
            if (carry == T(0)) {
                continue;
            }
            two_sum(sum, carry);
        }
        if (carry != T(0)) {
            sum += carry;
        }
        primary = sum;
        return sum;
    }

private:
    inline std::size_t compensation_offset(std::size_t bin, std::size_t i) const noexcept {
        if constexpr (Layout == accumulator_layout::soa) {
            return bin * size_ + i;
        }
        return i * compensation_terms_ + bin;
    }

    std::size_t size_ = 0;
    std::size_t compensation_terms_ = 0;
    T *compensation_block_ = nullptr;
};

}  // namespace compensated_blas
