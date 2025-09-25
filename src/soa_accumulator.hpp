#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory_resource>

namespace reidblas {

// Struct-of-array accumulator with configurable compensation bins per entry.
// Allocates storage from the supplied polymorphic memory resource.
template <typename T>
class soa_accumulator_t {
public:
    soa_accumulator_t(std::size_t length,
                      std::size_t compensation_terms,
                      std::pmr::memory_resource *resource = std::pmr::get_default_resource())
        : size_(length),
          compensation_terms_(compensation_terms),
          resource_(resource ? resource : std::pmr::get_default_resource()) {
        allocate_storage();
        zero_storage();
    }

    soa_accumulator_t(const soa_accumulator_t &) = delete;
    soa_accumulator_t &operator=(const soa_accumulator_t &) = delete;

    soa_accumulator_t(soa_accumulator_t &&other) noexcept
        : size_(other.size_),
          compensation_terms_(other.compensation_terms_),
          resource_(other.resource_),
          orig_(other.orig_),
          compensation_block_(other.compensation_block_),
          compensation_ptrs_(other.compensation_ptrs_),
          owns_storage_(other.owns_storage_) {
        other.reset_pointers();
    }

    soa_accumulator_t &operator=(soa_accumulator_t &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        release_storage();
        size_ = other.size_;
        compensation_terms_ = other.compensation_terms_;
        resource_ = other.resource_;
        orig_ = other.orig_;
        compensation_block_ = other.compensation_block_;
        compensation_ptrs_ = other.compensation_ptrs_;
        owns_storage_ = other.owns_storage_;
        other.reset_pointers();
        return *this;
    }

    ~soa_accumulator_t() { release_storage(); }

    inline std::size_t size() const noexcept { return size_; }
    inline std::size_t compensation_terms() const noexcept { return compensation_terms_; }

    inline T *data() noexcept { return orig_; }
    inline const T *data() const noexcept { return orig_; }

    inline T value(std::size_t i) const {
        assert(i < size_);
        return orig_[i];
    }

    inline T compensation(std::size_t bin, std::size_t i) const {
        assert(bin < compensation_terms_);
        assert(i < size_);
        return compensation_ptrs_[bin][i];
    }

    inline void accumulate(std::size_t i, const T &val) {
        assert(i < size_);
        if (compensation_terms_ == 0) {
            orig_[i] += val;
            return;
        }

        T carry = val;
        two_sum_in_place(orig_[i], carry);
        for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
            if (carry == T(0)) {
                return;
            }
            two_sum_in_place(compensation_ptrs_[bin][i], carry);
        }
        if (carry != T(0)) {
            orig_[i] += carry;
        }
    }

    inline T round(std::size_t i) {
        assert(i < size_);
        if (compensation_terms_ == 0) {
            return orig_[i];
        }

        T sum = orig_[i];
        T carry = T(0);
        for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
            const T value = compensation_ptrs_[bin][i] + carry;
            compensation_ptrs_[bin][i] = T(0);
            carry = value;
            if (carry == T(0)) {
                continue;
            }
            two_sum_in_place(sum, carry);
        }
        if (carry != T(0)) {
            sum += carry;
        }
        orig_[i] = sum;
        return sum;
    }

private:
    inline void allocate_storage() {
        if (!resource_) {
            resource_ = std::pmr::get_default_resource();
        }

        const std::size_t orig_bytes = size_ * sizeof(T);
        orig_ = size_ != 0 ? static_cast<T *>(resource_->allocate(orig_bytes, alignof(T))) : nullptr;

        const std::size_t compensation_count = compensation_terms_ * size_;
        const std::size_t compensation_bytes = compensation_count * sizeof(T);
        compensation_block_ = compensation_count != 0
                                  ? static_cast<T *>(resource_->allocate(compensation_bytes, alignof(T)))
                                  : nullptr;

        const std::size_t ptr_bytes = compensation_terms_ * sizeof(T *);
        compensation_ptrs_ = compensation_terms_ != 0
                                 ? static_cast<T **>(resource_->allocate(ptr_bytes, alignof(T *)))
                                 : nullptr;

        if (compensation_ptrs_) {
            for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
                compensation_ptrs_[bin] = compensation_block_ + bin * size_;
            }
        }
        owns_storage_ = true;
    }

    inline void release_storage() {
        if (!owns_storage_) {
            reset_pointers();
            return;
        }
        if (compensation_ptrs_) {
            resource_->deallocate(compensation_ptrs_, compensation_terms_ * sizeof(T *), alignof(T *));
        }
        if (compensation_block_) {
            resource_->deallocate(compensation_block_, compensation_terms_ * size_ * sizeof(T), alignof(T));
        }
        if (orig_) {
            resource_->deallocate(orig_, size_ * sizeof(T), alignof(T));
        }
        reset_pointers();
        resource_ = std::pmr::get_default_resource();
    }

    inline void reset_pointers() noexcept {
        size_ = 0;
        compensation_terms_ = 0;
        orig_ = nullptr;
        compensation_block_ = nullptr;
        compensation_ptrs_ = nullptr;
        owns_storage_ = false;
    }

    inline void zero_storage() {
        if (orig_) {
            std::fill_n(orig_, size_, T(0));
        }
        for (std::size_t bin = 0; bin < compensation_terms_; ++bin) {
            std::fill_n(compensation_ptrs_[bin], size_, T(0));
        }
    }

    static inline void two_sum_in_place(T &sum, T &value) {
        const T temp = sum + value;
        const T bp = temp - sum;
        const T error = (sum - (temp - bp)) + (value - bp);
        sum = temp;
        value = error;
    }

    std::size_t size_ = 0;
    std::size_t compensation_terms_ = 0;
    std::pmr::memory_resource *resource_ = std::pmr::get_default_resource();
    T *orig_ = nullptr;
    T *compensation_block_ = nullptr;
    T **compensation_ptrs_ = nullptr;
    bool owns_storage_ = false;
};

}  // namespace reidblas

