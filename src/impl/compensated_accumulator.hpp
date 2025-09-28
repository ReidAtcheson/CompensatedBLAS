#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory_resource>

#include "compensated_arithmetic.hpp"

namespace compensated_blas {

enum class accumulator_layout { soa, aos };

// Accumulator with configurable compensation bins per entry.
// Storage layout can be structure-of-arrays (default) or array-of-structures.
template <typename T, accumulator_layout Layout = accumulator_layout::soa>
class compensated_accumulator_t {
public:
    compensated_accumulator_t(std::size_t length,
                              std::size_t compensation_terms,
                              std::pmr::memory_resource *resource = std::pmr::get_default_resource())
        : size_(length),
          compensation_terms_(compensation_terms),
          resource_(resource ? resource : std::pmr::get_default_resource()) {
        allocate_storage();
        zero_storage();
    }

    compensated_accumulator_t(const compensated_accumulator_t &) = delete;
    compensated_accumulator_t &operator=(const compensated_accumulator_t &) = delete;

    compensated_accumulator_t(compensated_accumulator_t &&other) noexcept
        : size_(other.size_),
          compensation_terms_(other.compensation_terms_),
          resource_(other.resource_),
          compensation_block_(other.compensation_block_),
          owns_storage_(other.owns_storage_) {
        other.reset_pointers();
    }

    compensated_accumulator_t &operator=(compensated_accumulator_t &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        release_storage();
        size_ = other.size_;
        compensation_terms_ = other.compensation_terms_;
        resource_ = other.resource_;
        compensation_block_ = other.compensation_block_;
        owns_storage_ = other.owns_storage_;
        other.reset_pointers();
        return *this;
    }

    ~compensated_accumulator_t() { release_storage(); }

    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t compensation_terms() const noexcept { return compensation_terms_; }

    inline T compensation(std::size_t bin, std::size_t i) const {
        assert(bin < compensation_terms_);
        assert(i < size_);
        return compensation_block_[compensation_offset(bin, i)];
    }

    inline void accumulate(std::size_t i, T &primary, const T &val) {
        assert(i < size_);
        if (compensation_terms_ == 0) {
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
        if (compensation_terms_ == 0) {
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
    inline void allocate_storage() {
        if (!resource_) {
            resource_ = std::pmr::get_default_resource();
        }

        const std::size_t compensation_count = compensation_terms_ * size_;
        const std::size_t compensation_bytes = compensation_count * sizeof(T);
        compensation_block_ = compensation_count != 0
                                  ? static_cast<T *>(resource_->allocate(compensation_bytes, alignof(T)))
                                  : nullptr;
        owns_storage_ = true;
    }

    inline void release_storage() {
        if (!owns_storage_) {
            reset_pointers();
            return;
        }
        if (compensation_block_) {
            resource_->deallocate(compensation_block_, compensation_terms_ * size_ * sizeof(T), alignof(T));
        }
        reset_pointers();
        resource_ = std::pmr::get_default_resource();
    }

    inline void reset_pointers() noexcept {
        size_ = 0;
        compensation_terms_ = 0;
        compensation_block_ = nullptr;
        owns_storage_ = false;
    }

    inline void zero_storage() {
        if (compensation_block_) {
            std::fill_n(compensation_block_, compensation_terms_ * size_, T(0));
        }
    }

    inline std::size_t compensation_offset(std::size_t bin, std::size_t i) const noexcept {
        if constexpr (Layout == accumulator_layout::soa) {
            return bin * size_ + i;
        }
        return i * compensation_terms_ + bin;
    }

    std::size_t size_ = 0;
    std::size_t compensation_terms_ = 0;
    std::pmr::memory_resource *resource_ = std::pmr::get_default_resource();
    T *compensation_block_ = nullptr;
    bool owns_storage_ = false;
};

}  // namespace compensated_blas
