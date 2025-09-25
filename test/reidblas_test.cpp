#include <gtest/gtest.h>

#include "reidblas.hpp"

TEST(ReidblasSanity, IdentityReturnsInput) {
    constexpr reidblas::index_t value = 123;
    EXPECT_EQ(reidblas::identity(value), value);
}
