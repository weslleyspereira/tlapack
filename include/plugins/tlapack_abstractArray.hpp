/// @file tlapack_abstractArray.hpp
/// @author Weslley S Pereira, University of Colorado Denver, US
///
/// This file contains has two purposes:
///     1. It serves as a template for writing <T>LAPACK abstract arrays.
///     2. It is used by Doxygen to generate the documentation.
//
// Copyright (c) 2021-2022, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __TLAPACK_ABSTRACT_ARRAY_HH__
#define __TLAPACK_ABSTRACT_ARRAY_HH__

#include "blas/arrayTraits.hpp"

namespace blas {

    // -------------------------------------------------------------------------
    // Data descriptors for matrices in <T>LAPACK
    
    /**
     * @brief Return the number of rows of a given matrix.
     * 
     * @tparam idx_t    Index type.
     * @tparam matrix_t Matrix type.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class idx_t, class matrix_t >
    inline constexpr idx_t
    nrows( const matrix_t& x );

    /**
     * @brief Return the number of columns of a given matrix.
     * 
     * @tparam idx_t    Index type.
     * @tparam matrix_t Matrix type.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class idx_t, class matrix_t >
    inline constexpr idx_t
    ncols( const matrix_t& A );

    /**
     * @brief Read policy.
     * 
     * Defines the pairs (i,j) where A(i,j) returns a valid value.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam access_t Access type. One of the following:
     *      MatrixAccessPolicy
     *      dense_t,
     *      upperHessenberg_t,
     *      lowerHessenberg_t,
     *      upperTriangle_t,
     *      lowerTriangle_t,
     *      strictUpper_t,
     *      strictLower_t,
     *      band_t.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class access_t, class matrix_t >
    inline constexpr access_t
    read_policy( const matrix_t& A );

    /**
     * @brief Write policy.
     * 
     * Defines the pairs (i,j) where A(i,j) returns a valid reference for
     * reading and writing.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam access_t Access type. One of the following:
     *      MatrixAccessPolicy
     *      dense_t,
     *      upperHessenberg_t,
     *      lowerHessenberg_t,
     *      upperTriangle_t,
     *      lowerTriangle_t,
     *      strictUpper_t,
     *      strictLower_t,
     *      band_t.
     * 
     * @param A Matrix.
     * 
     * @ingroup utils
     */
    template< class access_t, class matrix_t >
    inline constexpr access_t
    write_policy( const matrix_t& A );

    // -------------------------------------------------------------------------
    // Data descriptors for vectors in <T>LAPACK

    /**
     * @brief Return the number of elements of a given vector.
     * 
     * @tparam idx_t    Index type.
     * @tparam vector_t Vector type.
     * 
     * @param v Vector.
     * 
     * @ingroup utils
     */
    template< class idx_t, class vector_t >
    inline constexpr idx_t
    size( const vector_t& v );

    // -------------------------------------------------------------------------
    // Block operations with matrices in <T>LAPACK

    /**
     * @brief Extracts a submatrix from a given matrix.
     * 
     * Note that a submatrix is also a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * @param cols  Pair (j,l).
     *      j   is the index of the first column, and
     *      l-1 is the index of the last column.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    submatrix( const matrix_t& A, pair_t&& rows, pair_t&& cols );
    
    /**
     * @brief Extracts a set of rows from a given matrix.
     * 
     * @note A set of rows is also a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    rows( const matrix_t& A, pair_t&& rows );
    
    /**
     * @brief Extracts a set of columns from a given matrix.
     * 
     * @note A set of columns is also a matrix.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param A     Matrix.
     * @param cols  Pair (j,l).
     *      j   is the index of the first column, and
     *      l-1 is the index of the last column.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class pair_t >
    inline constexpr auto
    cols( const matrix_t& A, pair_t&& cols );
    
    /**
     * @brief Extracts a row from a given matrix.
     * 
     * @note A row is treated as a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix.
     * @param rowIdx    Row index.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    row( const matrix_t& A, idx_t rowIdx );
    
    /**
     * @brief Extracts a column from a given matrix.
     * 
     * @note A column is treated as a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix.
     * @param colIdx    Column index.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    col( const matrix_t& A, idx_t colIdx );

    /**
     * @brief Extracts a diagonal from a given matrix. 
     * 
     * @note A diagonal is treated as a vector.
     * 
     * @tparam matrix_t Matrix type.
     * @tparam idx_t    Index type.
     * 
     * @param A         Matrix of size m-by-n. 
     * @param diagIdx   Diagonal index.
     *      - diagIdx = 0: main diagonal.
     *          Vector of size min( m, n ).
     *      - diagIdx < 0: subdiagonal starting on A( -diagIdx, 0 ).
     *          Vector of size min( m, n-diagIdx ) + diagIdx.
     *      - diagIdx > 0: superdiagonal starting on A( 0, diagIdx ).
     *          Vector of size min( m+diagIdx, n ) - diagIdx.
     * 
     * @ingroup utils
     */
    template< class matrix_t, class idx_t >
    inline constexpr auto
    diag( const matrix_t& A, idx_t diagIdx = 0 );

    /**
     * @brief Interpret a given vector as a 1-column matrix.
     * 
     * @tparam vector_t Vector type.
     * 
     * @param v Vector of size n.
     * 
     * @return Matrix of size n-by-1.
     */
    template< class vector_t >
    inline constexpr
    auto
    interpretAsMatrix( const vector_t& v );

    /// @returns The input because it is already a matrix.
    template< class matrix_t >
    inline constexpr
    const auto&
    interpretAsMatrix( const matrix_t& A )
    {
        return A;
    }

    // -------------------------------------------------------------------------
    // Block operations with vectors in <T>LAPACK

    /**
     * @brief Extracts a subvector from a vector.
     * 
     * @note A subvector is also a vector.
     * 
     * @tparam vector_t Vector type.
     * @tparam pair_t   Pair of integers.
     *      Stored in pair::first and pair::second.
     * 
     * @param v     Vector.
     * @param rows  Pair (i,k).
     *      i   is the index of the first row, and
     *      k-1 is the index of the last row.
     * 
     * @ingroup utils
     */
    template< class vector_t, class pair_t >
    inline constexpr auto
    subvector( const vector_t& v, pair_t&& rows );

} // namespace blas

namespace lapack {

    // Data descriptors for matrices in <T>LAPACK
    using blas::nrows;
    using blas::ncols;
    using blas::read_policy;
    using blas::write_policy;

    // Data descriptors for vectors in <T>LAPACK
    using blas::size;

    // Block operations with matrices in <T>LAPACK
    using blas::submatrix;
    using blas::rows;
    using blas::cols;
    using blas::row;
    using blas::col;
    using blas::diag;

    // Block operations with vectors in <T>LAPACK
    using blas::subvector;

} // namespace lapack

#endif // __TLAPACK_ABSTRACT_ARRAY_HH__