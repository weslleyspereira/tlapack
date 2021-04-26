// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "types.hpp"
#include "exception.hpp"
#include "utils.hpp"
#include "constants.hpp"

#include <limits>

namespace blas {

// =============================================================================
/// @return Index of infinity-norm of vector, $|| x ||_{inf}$,
///     $\text{argmax}_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|$.
/// Returns -1 if n = 0.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup iamax

template< typename T >
size_t iamax(
    size_t n,
    T const *x, int_t incx )
{
    typedef real_type<T> real_t;
    
    bool waitForNaNcheck = true;

    if (incx == 1) {
        // unit stride
        #pragma omp parallel sections num_threads(2) shared(waitForNaNcheck)
        {
            #pragma omp section
            { // Check for nans
                for (size_t i = 0; i < n; ++i) {
                    if ( isnan(x[i]) ) {
                        // return when first NaN found
                        return i;
                    }
                }
                waitForNaNcheck = false;
            }
            #pragma omp section
            { // Regular section + check for infs

                bool scaledsmax = false; // indicates whether sx(i) finite but abs(real(sx(i))) + abs(imag(sx(i))) = Inf
                real_t smax = -1;
                size_t index = INVALID_INDEX;
                const real_t oneFourth = 0.25;

                for (size_t i = 0; i < n; ++i) {
                    if ( isinf(x[i]) ) {
                        // record location of first Inf
                        index = i;
                        break;
                    }
                    else { // still no Inf found yet
                        if ( !scaledsmax ) { // no abs(real(sx(i))) + abs(imag(sx(i))) = Inf  yet
                            real_t a = abs1(x[i]);
                            if ( isinf(a) ) {
                                scaledsmax = true;
                                smax = abs1( oneFourth*x[i] );
                                index = i;
                            }
                            else if ( a > smax ) { // and everything finite so far
                                smax = a;
                                index = i;
                            }
                        }
                        else { // scaledsmax = true
                            real_t a = abs1( oneFourth*x[i] );
                            if ( a > smax ) { // and everything finite so far
                                smax = a;
                                index = i;
                            }
                        }
                    }
                }
                
                while ( waitForNaNcheck ) {
                    // It should never be stuck here, since check for nans is faster than the regular iamax section
                    #pragma omp flush(waitForNaNcheck)
                }
                return index;
            }
        }
    }
    else {
        // non-unit stride
        #pragma omp parallel sections num_threads(2) shared(waitForNaNcheck)
        {
            #pragma omp section
            { // Check for nans
                int_t ix = 0;
                for (size_t i = 0; i < n; ++i) {
                    if ( isnan(x[ix]) ) {
                        // return when first NaN found
                        return i;
                    }
                    ix += incx;
                }
                waitForNaNcheck = false;
            }
            #pragma omp section
            { // Regular section + check for infs

                bool scaledsmax = false; // indicates whether sx(i) finite but abs(real(sx(i))) + abs(imag(sx(i))) = Inf
                real_t smax = -1;
                size_t index = INVALID_INDEX;
                const real_t oneFourth = 0.25;

                int_t ix = 0;
                for (size_t i = 0; i < n; ++i) {
                    if ( isinf(x[ix]) ) {
                        // record location of first Inf
                        index = i;
                        break;
                    }
                    else { // still no Inf found yet
                        if ( !scaledsmax ) { // no abs(real(sx(i))) + abs(imag(sx(i))) = Inf  yet
                            real_t a = abs1(x[ix]);
                            if ( isinf(a) ) {
                                scaledsmax = true;
                                smax = abs1( oneFourth*x[ix] );
                                index = i;
                            }
                            else if ( a > smax ) { // and everything finite so far
                                smax = a;
                                index = i;
                            }
                        }
                        else { // scaledsmax = true
                            real_t a = abs1( oneFourth*x[ix] );
                            if ( a > smax ) { // and everything finite so far
                                smax = a;
                                index = i;
                            }
                        }
                    }
                    ix += incx;
                }

                while ( waitForNaNcheck ) {
                    // It should never be stuck here, since check for nans is faster than the regular iamax section
                    #pragma omp flush(waitForNaNcheck)
                }
                return index;
            }
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_IAMAX_HH
