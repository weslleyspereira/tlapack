// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_ONECOLMATRIX_HH__
#define __TLAPACK_ONECOLMATRIX_HH__

#include "base/arrayTraits.hpp"

namespace tlapack {

    template< class vector_t >
    struct oneColMatrix : public vector_t {
        
        using idx_t = size_type< vector_t >;
        using T     = type_t< vector_t >;

        inline constexpr T&
        operator()( idx_t i, idx_t j ){ return (*this)[i]; }
        
        inline constexpr T
        operator()( idx_t i, idx_t j ) const { return (*this)[i]; }
    };

    // Layout
    template< class vector_t >
    constexpr Layout layout< oneColMatrix<vector_t> > = Layout::Unspecified;

    // Number of rows
    template< class vector_t >
    inline constexpr auto
    nrows( const oneColMatrix<vector_t>& A ){ return size(A); }

    // Number of columns
    template< class vector_t >
    inline constexpr auto
    ncols( const oneColMatrix<vector_t>& A ){ return 1; }

} // namespace tlapack

#endif // __TLAPACK_EIGEN_HH__
