#pragma once

/// \file valueTypes.h

#include <cassert>
#include <sstream>
#include <string>

#include <cudaExperiments/math.h>

/// \class Vec3f
///
/// 3-element floating point vector class.
/// Useful for encoding vectors and points.
class alignas( 16 ) Vec3f
{
public:
    /// Element read-access.
    inline __host__ __device__ const float& operator[]( size_t i_index ) const
    {
        return m_values[ i_index ];
    }

    /// Element write-access.
    inline __host__ __device__ float& operator[]( size_t i_index )
    {
        return m_values[ i_index ];
    }

private:
    float m_values[ 3 ] = {0};
    float m_padding[ 1 ] = {0};
};

static_assert( sizeof( Vec3f ) == 16 );

/// \class Mat4f
///
/// 4x4 floating point value matrix class.
/// Minimally implemented to use for experiments.
class Mat4f
{
public:
    template < typename... ValuesT >
    __host__ __device__ Mat4f( ValuesT... i_values )
        : m_values{i_values...}
    {
    }

    /// Get the identity matrix.
    ///
    /// \return identity matrix.
    static Mat4f Identity()
    {
        return Mat4f(
            {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f} );
    }

    /// Matrix element const accessor.
    inline __host__ __device__ const float& operator()( size_t i_row, size_t i_column ) const
    {
        return m_values[ i_row * 4 + i_column ];
    }

    /// Matrix element mutable accessor.
    inline __host__ __device__ float& operator()( size_t i_row, size_t i_column )
    {
        return m_values[ i_row * 4 + i_column ];
    }

    /// Get the string representation of this matrix, for debugging purposes.
    inline std::string GetString() const
    {
        std::stringstream ss;
        ss << "Mat4f( ";
        for ( size_t row = 0; row < 4; ++row )
        {
            for ( size_t col = 0; col < 4; ++col )
            {
                ss << ( *this )( row, col );
                if ( !( row == 3 && col == 3 ) )
                {
                    ss << ", ";
                }
            }
        }
        ss << " )";
        return ss.str();
    }

    /// Equality operator.
    inline __host__ __device__ bool operator==( const Mat4f& i_matrix ) const
    {
        return AlmostEqual( m_values[ 0 ], i_matrix.m_values[ 0 ] ) &&
               AlmostEqual( m_values[ 1 ], i_matrix.m_values[ 1 ] ) &&
               AlmostEqual( m_values[ 2 ], i_matrix.m_values[ 2 ] ) &&
               AlmostEqual( m_values[ 3 ], i_matrix.m_values[ 3 ] ) &&
               AlmostEqual( m_values[ 4 ], i_matrix.m_values[ 4 ] ) &&
               AlmostEqual( m_values[ 5 ], i_matrix.m_values[ 5 ] ) &&
               AlmostEqual( m_values[ 6 ], i_matrix.m_values[ 6 ] ) &&
               AlmostEqual( m_values[ 7 ], i_matrix.m_values[ 7 ] ) &&
               AlmostEqual( m_values[ 8 ], i_matrix.m_values[ 8 ] ) &&
               AlmostEqual( m_values[ 9 ], i_matrix.m_values[ 9 ] ) &&
               AlmostEqual( m_values[ 10 ], i_matrix.m_values[ 10 ] ) &&
               AlmostEqual( m_values[ 11 ], i_matrix.m_values[ 11 ] ) &&
               AlmostEqual( m_values[ 12 ], i_matrix.m_values[ 12 ] ) &&
               AlmostEqual( m_values[ 13 ], i_matrix.m_values[ 13 ] ) &&
               AlmostEqual( m_values[ 14 ], i_matrix.m_values[ 14 ] ) &&
               AlmostEqual( m_values[ 15 ], i_matrix.m_values[ 15 ] );
    }

    inline __host__ __device__ bool operator!=( const Mat4f& i_matrix ) const
    {
        return !( ( *this ) == i_matrix );
    }

private:
    float m_values[ 16 ] = {0};
};

static_assert( sizeof( Vec3f ) == 16 );

/// Compute the product of two matrices: \p i_lhs * \p i_rhs.
///
/// \param i_lhs left hand side matrix.
/// \param i_rhs right hand side matrix.
///
/// \return the matrix product.
inline __host__ __device__ Mat4f operator*( const Mat4f& i_lhs, const Mat4f& i_rhs )
{
    Mat4f matrix;
    matrix( 0, 0 ) = i_lhs( 0, 0 ) * i_rhs( 0, 0 ) + i_lhs( 0, 1 ) * i_rhs( 1, 0 ) + i_lhs( 0, 2 ) * i_rhs( 2, 0 ) +
                     i_lhs( 0, 3 ) * i_rhs( 3, 0 );
    matrix( 0, 1 ) = i_lhs( 0, 0 ) * i_rhs( 0, 1 ) + i_lhs( 0, 1 ) * i_rhs( 1, 1 ) + i_lhs( 0, 2 ) * i_rhs( 2, 1 ) +
                     i_lhs( 0, 3 ) * i_rhs( 3, 1 );
    matrix( 0, 2 ) = i_lhs( 0, 0 ) * i_rhs( 0, 2 ) + i_lhs( 0, 1 ) * i_rhs( 1, 2 ) + i_lhs( 0, 2 ) * i_rhs( 2, 2 ) +
                     i_lhs( 0, 3 ) * i_rhs( 3, 2 );
    matrix( 0, 3 ) = i_lhs( 0, 0 ) * i_rhs( 0, 3 ) + i_lhs( 0, 1 ) * i_rhs( 1, 3 ) + i_lhs( 0, 2 ) * i_rhs( 2, 3 ) +
                     i_lhs( 0, 3 ) * i_rhs( 3, 3 );
    matrix( 1, 0 ) = i_lhs( 1, 0 ) * i_rhs( 0, 0 ) + i_lhs( 1, 1 ) * i_rhs( 1, 0 ) + i_lhs( 1, 2 ) * i_rhs( 2, 0 ) +
                     i_lhs( 1, 3 ) * i_rhs( 3, 0 );
    matrix( 1, 1 ) = i_lhs( 1, 0 ) * i_rhs( 0, 1 ) + i_lhs( 1, 1 ) * i_rhs( 1, 1 ) + i_lhs( 1, 2 ) * i_rhs( 2, 1 ) +
                     i_lhs( 1, 3 ) * i_rhs( 3, 1 );
    matrix( 1, 2 ) = i_lhs( 1, 0 ) * i_rhs( 0, 2 ) + i_lhs( 1, 1 ) * i_rhs( 1, 2 ) + i_lhs( 1, 2 ) * i_rhs( 2, 2 ) +
                     i_lhs( 1, 3 ) * i_rhs( 3, 2 );
    matrix( 1, 3 ) = i_lhs( 1, 0 ) * i_rhs( 0, 3 ) + i_lhs( 1, 1 ) * i_rhs( 1, 3 ) + i_lhs( 1, 2 ) * i_rhs( 2, 3 ) +
                     i_lhs( 1, 3 ) * i_rhs( 3, 3 );
    matrix( 2, 0 ) = i_lhs( 2, 0 ) * i_rhs( 0, 0 ) + i_lhs( 2, 1 ) * i_rhs( 1, 0 ) + i_lhs( 2, 2 ) * i_rhs( 2, 0 ) +
                     i_lhs( 2, 3 ) * i_rhs( 3, 0 );
    matrix( 2, 1 ) = i_lhs( 2, 0 ) * i_rhs( 0, 1 ) + i_lhs( 2, 1 ) * i_rhs( 1, 1 ) + i_lhs( 2, 2 ) * i_rhs( 2, 1 ) +
                     i_lhs( 2, 3 ) * i_rhs( 3, 1 );
    matrix( 2, 2 ) = i_lhs( 2, 0 ) * i_rhs( 0, 2 ) + i_lhs( 2, 1 ) * i_rhs( 1, 2 ) + i_lhs( 2, 2 ) * i_rhs( 2, 2 ) +
                     i_lhs( 2, 3 ) * i_rhs( 3, 2 );
    matrix( 2, 3 ) = i_lhs( 2, 0 ) * i_rhs( 0, 3 ) + i_lhs( 2, 1 ) * i_rhs( 1, 3 ) + i_lhs( 2, 2 ) * i_rhs( 2, 3 ) +
                     i_lhs( 2, 3 ) * i_rhs( 3, 3 );
    matrix( 3, 0 ) = i_lhs( 3, 0 ) * i_rhs( 0, 0 ) + i_lhs( 3, 1 ) * i_rhs( 1, 0 ) + i_lhs( 3, 2 ) * i_rhs( 2, 0 ) +
                     i_lhs( 3, 3 ) * i_rhs( 3, 0 );
    matrix( 3, 1 ) = i_lhs( 3, 0 ) * i_rhs( 0, 1 ) + i_lhs( 3, 1 ) * i_rhs( 1, 1 ) + i_lhs( 3, 2 ) * i_rhs( 2, 1 ) +
                     i_lhs( 3, 3 ) * i_rhs( 3, 1 );
    matrix( 3, 2 ) = i_lhs( 3, 0 ) * i_rhs( 0, 2 ) + i_lhs( 3, 1 ) * i_rhs( 1, 2 ) + i_lhs( 3, 2 ) * i_rhs( 2, 2 ) +
                     i_lhs( 3, 3 ) * i_rhs( 3, 2 );
    matrix( 3, 3 ) = i_lhs( 3, 0 ) * i_rhs( 0, 3 ) + i_lhs( 3, 1 ) * i_rhs( 1, 3 ) + i_lhs( 3, 2 ) * i_rhs( 2, 3 ) +
                     i_lhs( 3, 3 ) * i_rhs( 3, 3 );
    return matrix;
}

