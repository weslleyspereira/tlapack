/// @file base/StrongZero.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BASE_STRONGZERO_HH
#define TLAPACK_BASE_STRONGZERO_HH

#include <cstdint>

namespace tlapack {

/**
 * @brief Auxiliary data type
 *
 * Suppose x is of type T. Then:
 *
 * 1. T(StrongZero()) is equivalent to T(0).
 * 2. x *= StrongZero() is equivalent to x = T(0).
 * 3. x += StrongZero() does not modify x.
 *
 * This class satisfies:
 *
 *      x * StrongZero() = StrongZero()
 *      StrongZero() * x = StrongZero()
 *      x + StrongZero() = x
 *      StrongZero() + x = x
 *
 */
struct StrongZero {
    template <typename T>
    explicit constexpr operator T() const
    {
        return T(0);
    }

    template <typename T>
    friend constexpr T& operator*=(T& lhs, const StrongZero&)
    {
        lhs = T(0);
        return lhs;
    }

    template <typename T>
    friend constexpr const StrongZero operator*(const StrongZero&, const T&)
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr const StrongZero operator*(const T&, const StrongZero&)
    {
        return StrongZero();
    }

    template <typename T>
    friend constexpr const T& operator+=(const T& lhs, const StrongZero&)
    {
        return lhs;
    }

    template <typename T>
    friend constexpr const T operator+(const StrongZero&, const T& rhs)
    {
        return rhs;
    }

    template <typename T>
    friend constexpr const T operator+(const T& lhs, const StrongZero&)
    {
        return lhs;
    }
};

// forward declarations
template <typename... Types>
struct scalar_type_traits;
template <typename... Types>
struct real_type_traits;
template <typename... Types>
struct complex_type_traits;

// for either StrongZero, return the other type
template <typename T>
struct scalar_type_traits<StrongZero, T> {
    using type = T;
};

// for either StrongZero, return the other type
template <typename T>
struct scalar_type_traits<T, StrongZero> {
    using type = T;
};

// for both StrongZero, return int8_t
template <>
struct scalar_type_traits<StrongZero, StrongZero> {
    using type = int8_t;
};

// for one StrongZero, real type is int8_t
template <>
struct real_type_traits<StrongZero> {
    using type = int8_t;
};

}  // namespace tlapack

#endif  // TLAPACK_BASE_STRONGZERO_HH
