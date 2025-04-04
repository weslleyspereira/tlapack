/// @file lascl.hpp Multiplies a matrix by a scalar.
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lascl.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LASCL_HH
#define TLAPACK_LASCL_HH

#include "tlapack/base/constants.hpp"
#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * @brief Multiplies a matrix A by the real scalar a/b.
 *
 * Multiplication of a matrix A by scalar a/b is done without over/underflow as
 * long as the final result $a A/b$ does not over/underflow.
 *
 * @tparam uplo_t Type of access inside the algorithm.
 *      Either Uplo or any type that implements
 *          operator Uplo().
 * @tparam matrix_t Matrix type.
 * @tparam a_type Type of the coefficient a.
 *      a_type cannot be a complex type.
 * @tparam b_type Type of the coefficient b.
 *      b_type cannot be a complex type.
 *
 * @param[in] uplo Determines the entries of A that are scaled by a/b.
 *      The following access types are allowed:
 *          Uplo::General,
 *          Uplo::UpperHessenberg,
 *          Uplo::LowerHessenberg,
 *          Uplo::Upper,
 *          Uplo::Lower,
 *          Uplo::StrictUpper,
 *          Uplo::StrictLower.
 *
 * @param[in] b The denominator of the scalar a/b.
 * @param[in] a The numerator of the scalar a/b.
 * @param[in,out] A Matrix to be scaled by a/b.
 *
 * @return  0 if success..
 *
 * @ingroup auxiliary
 */
template <TLAPACK_UPLO uplo_t,
          TLAPACK_MATRIX matrix_t,
          TLAPACK_REAL a_type,
          TLAPACK_REAL b_type,
          enable_if_t<(
                          /* Requires: */
                          is_real<a_type> && is_real<b_type>),
                      int> = 0>
int lascl(uplo_t uplo, const b_type& b, const a_type& a, matrix_t& A)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<a_type, b_type>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    // constants
    const real_t small = safe_min<real_t>();
    const real_t big = safe_max<real_t>();

    // check arguments
    tlapack_check_false(
        (uplo != Uplo::General) && (uplo != Uplo::UpperHessenberg) &&
        (uplo != Uplo::LowerHessenberg) && (uplo != Uplo::Upper) &&
        (uplo != Uplo::Lower) && (uplo != Uplo::StrictUpper) &&
        (uplo != Uplo::StrictLower));
    tlapack_check_false((b == b_type(0)) || isnan(b));
    tlapack_check_false(isnan(a));

    // quick return
    if (m <= 0 || n <= 0) return 0;

    bool done = false;
    real_t a_ = a, b_ = b;
    while (!done) {
        real_t c, a1, b1 = b * small;
        if (b1 == b_) {
            // b is not finite:
            //  c is a correctly signed zero if a is finite,
            //  c is NaN otherwise.
            c = a_ / b_;
            done = true;
        }
        else {  // b is finite
            a1 = a_ / big;
            if (a1 == a_) {
                // a is either 0 or an infinity number:
                //  in both cases, c = a serves as the correct multiplication
                //  factor.
                c = a_;
                done = true;
            }
            else if ((abs(b1) > abs(a_)) && (a_ != real_t(0))) {
                // a is a non-zero finite number and abs(a/b) < small:
                //  Set c = small as the multiplication factor,
                //  Multiply b by the small factor.
                c = small;
                done = false;
                b_ = b1;
            }
            else if (abs(a1) > abs(b_)) {
                // abs(a/b) > big:
                //  Set c = big as the multiplication factor,
                //  Divide a by the big factor.
                c = big;
                done = false;
                a_ = a1;
            }
            else {
                // small <= abs(a/b) <= big:
                //  Set c = a/b as the multiplication factor.
                c = a_ / b_;
                done = true;
            }
        }

        if (uplo == Uplo::UpperHessenberg) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j + 2 : m); ++i)
                    A(i, j) *= c;
        }
        else if (uplo == Uplo::Upper) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j + 1 : m); ++i)
                    A(i, j) *= c;
        }
        else if (uplo == Uplo::StrictUpper) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < ((j < m) ? j : m); ++i)
                    A(i, j) *= c;
        }
        else if (uplo == Uplo::LowerHessenberg) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = ((j > 1) ? j - 1 : 0); i < m; ++i)
                    A(i, j) *= c;
        }
        else if (uplo == Uplo::Lower) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j; i < m; ++i)
                    A(i, j) *= c;
        }
        else if (uplo == Uplo::StrictLower) {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j + 1; i < m; ++i)
                    A(i, j) *= c;
        }
        else  // if ( uplo == Uplo::General )
        {
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < m; ++i)
                    A(i, j) *= c;
        }
    }

    return 0;
}

/**
 * @brief Multiplies a matrix A by the real scalar a/b.
 *
 * Specific implementation for band access types.
 *
 * @param[in] accessType Determines the entries of A that are scaled by a/b.
 *
 * @param[in] b The denominator of the scalar a/b.
 * @param[in] a The numerator of the scalar a/b.
 * @param[in,out] A Matrix to be scaled by a/b.
 *
 * @see lascl(uplo_t uplo, const b_type& b, const a_type& a, matrix_t& A)
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_REAL a_type,
          TLAPACK_REAL b_type,
          enable_if_t<(
                          /* Requires: */
                          is_real<a_type> && is_real<b_type>),
                      int> = 0>
int lascl(BandAccess accessType, const b_type& b, const a_type& a, matrix_t& A)
{
    // data traits
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<a_type, b_type>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = accessType.lower_bandwidth;
    const idx_t ku = accessType.upper_bandwidth;

    // constants
    const real_t small = safe_min<real_t>();
    const real_t big = safe_max<real_t>();

    // check arguments
    tlapack_check_false((kl < 0) || (kl >= m) || (ku < 0) || (ku >= n));
    tlapack_check_false((b == b_type(0)) || isnan(b));
    tlapack_check_false(isnan(a));

    // quick return
    if (m <= 0 || n <= 0) return 0;

    bool done = false;
    real_t a_ = a, b_ = b;
    while (!done) {
        real_t c, a1, b1 = b * small;
        if (b1 == b_) {
            // b is not finite:
            //  c is a correctly signed zero if a is finite,
            //  c is NaN otherwise.
            c = a_ / b_;
            done = true;
        }
        else {  // b is finite
            a1 = a_ / big;
            if (a1 == a_) {
                // a is either 0 or an infinity number:
                //  in both cases, c = a serves as the correct multiplication
                //  factor.
                c = a_;
                done = true;
            }
            else if ((abs(b1) > abs(a_)) && (a_ != real_t(0))) {
                // a is a non-zero finite number and abs(a/b) < small:
                //  Set c = small as the multiplication factor,
                //  Multiply b by the small factor.
                c = small;
                done = false;
                b_ = b1;
            }
            else if (abs(a1) > abs(b_)) {
                // abs(a/b) > big:
                //  Set c = big as the multiplication factor,
                //  Divide a by the big factor.
                c = big;
                done = false;
                a_ = a1;
            }
            else {
                // small <= abs(a/b) <= big:
                //  Set c = a/b as the multiplication factor.
                c = a_ / b_;
                done = true;
            }
        }

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = ((j >= ku) ? (j - ku) : 0); i < min(m, j + kl + 1);
                 ++i)
                A(i, j) *= c;
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LASCL_HH
