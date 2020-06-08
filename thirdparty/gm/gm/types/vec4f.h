//
// This file is auto-generated, please do not modify directly!
//

#pragma once

/// \file vec4f.h

#include <cmath>
#include <cstring>
#include <sstream>

#include <gm/base/almost.h>
#include <gm/base/assert.h>
#include <gm/gm.h>

GM_NS_OPEN

/// \class Vec4f
class Vec4f final
{
public:
    /// Type of \ref Vec4f's elements.
    using ElementType = float;

    /// Default constructor.
    Vec4f() = default;

    /// Destructor.
    ~Vec4f() = default;

    /// Element-wise constructor.
    explicit Vec4f( const float& i_element0, const float& i_element1, const float& i_element2, const float& i_element3 )
        : m_elements{i_element0, i_element1, i_element2, i_element3}
    {
        GM_ASSERT( !HasNans() );
    }

#ifdef GM_DEBUG
    /// Copy constructor.
    Vec4f( const Vec4f& i_vector )
    {
        std::memcpy( ( void* ) m_elements, ( const void* ) i_vector.m_elements, sizeof( float ) * 4 );
        GM_ASSERT( !HasNans() );
    }

    /// Copy assignment operator.
    Vec4f& operator=( const Vec4f& i_vector )
    {
        std::memcpy( ( void* ) m_elements, ( const void* ) i_vector.m_elements, sizeof( float ) * 4 );
        GM_ASSERT( !HasNans() );
        return *this;
    }
#endif

    /// Element-wise index read accessor.
    GM_HOST_DEVICE inline float& operator[]( size_t i_index )
    {
        GM_ASSERT( !HasNans() );
        GM_ASSERT( i_index < 4 );
        return m_elements[ i_index ];
    }

    /// Element-wise index write accessor.
    GM_HOST_DEVICE inline const float& operator[]( size_t i_index ) const
    {
        GM_ASSERT( !HasNans() );
        GM_ASSERT( i_index < 4 );
        return m_elements[ i_index ];
    }

    //
    // Arithmetic Operator Overloading.
    //

    /// Vector addition.
    GM_HOST_DEVICE inline Vec4f operator+( const Vec4f& i_vector ) const
    {
        GM_ASSERT( !HasNans() );
        return Vec4f( m_elements[ 0 ] + i_vector.m_elements[ 0 ],
                      m_elements[ 1 ] + i_vector.m_elements[ 1 ],
                      m_elements[ 2 ] + i_vector.m_elements[ 2 ],
                      m_elements[ 3 ] + i_vector.m_elements[ 3 ] );
    }

    /// Vector addition assignment.
    GM_HOST_DEVICE inline Vec4f& operator+=( const Vec4f& i_vector )
    {
        GM_ASSERT( !HasNans() );
        m_elements[ 0 ] += i_vector.m_elements[ 0 ];
        m_elements[ 1 ] += i_vector.m_elements[ 1 ];
        m_elements[ 2 ] += i_vector.m_elements[ 2 ];
        m_elements[ 3 ] += i_vector.m_elements[ 3 ];
        return *this;
    }

    /// Vector subtraction.
    GM_HOST_DEVICE inline Vec4f operator-( const Vec4f& i_vector ) const
    {
        GM_ASSERT( !HasNans() );
        return Vec4f( m_elements[ 0 ] - i_vector.m_elements[ 0 ],
                      m_elements[ 1 ] - i_vector.m_elements[ 1 ],
                      m_elements[ 2 ] - i_vector.m_elements[ 2 ],
                      m_elements[ 3 ] - i_vector.m_elements[ 3 ] );
    }

    /// Vector subtraction assignment.
    GM_HOST_DEVICE inline Vec4f& operator-=( const Vec4f& i_vector )
    {
        GM_ASSERT( !HasNans() );
        m_elements[ 0 ] -= i_vector.m_elements[ 0 ];
        m_elements[ 1 ] -= i_vector.m_elements[ 1 ];
        m_elements[ 2 ] -= i_vector.m_elements[ 2 ];
        m_elements[ 3 ] -= i_vector.m_elements[ 3 ];
        return *this;
    }

    /// Scalar multiplication assignment.
    GM_HOST_DEVICE inline Vec4f& operator*=( const float& i_scalar )
    {
        GM_ASSERT( !HasNans() );
        m_elements[ 0 ] *= i_scalar;
        m_elements[ 1 ] *= i_scalar;
        m_elements[ 2 ] *= i_scalar;
        m_elements[ 3 ] *= i_scalar;
        return *this;
    }

    /// Scalar division.
    GM_HOST_DEVICE inline Vec4f operator/( const float& i_scalar ) const
    {
        GM_ASSERT( !HasNans() );
        GM_ASSERT( i_scalar != 0.0f );
        return Vec4f( m_elements[ 0 ] / i_scalar,
                      m_elements[ 1 ] / i_scalar,
                      m_elements[ 2 ] / i_scalar,
                      m_elements[ 3 ] / i_scalar );
    }

    /// Scalar division assignment.
    GM_HOST_DEVICE inline Vec4f& operator/=( const float& i_scalar )
    {
        GM_ASSERT( !HasNans() );
        GM_ASSERT( i_scalar != 0.0f );
        m_elements[ 0 ] /= i_scalar;
        m_elements[ 1 ] /= i_scalar;
        m_elements[ 2 ] /= i_scalar;
        m_elements[ 3 ] /= i_scalar;
        return *this;
    }

    /// Unary negation.
    GM_HOST_DEVICE inline Vec4f operator-() const
    {
        GM_ASSERT( !HasNans() );
        return Vec4f( -m_elements[ 0 ], -m_elements[ 1 ], -m_elements[ 2 ], -m_elements[ 3 ] );
    }

    /// X component accessor for the first element.
    GM_HOST_DEVICE inline float X() const
    {
        GM_ASSERT( !HasNans() );
        return m_elements[ 0 ];
    }

    /// Y component accessor for the second element.
    GM_HOST_DEVICE inline float Y() const
    {
        GM_ASSERT( !HasNans() );
        return m_elements[ 1 ];
    }

    /// Z component accessor for the third element.
    GM_HOST_DEVICE inline float Z() const
    {
        GM_ASSERT( !HasNans() );
        return m_elements[ 2 ];
    }

    /// W component accessor for the fourth element.
    GM_HOST_DEVICE inline float W() const
    {
        GM_ASSERT( !HasNans() );
        return m_elements[ 3 ];
    }

    /// Comparison operator
    GM_HOST_DEVICE inline bool operator==( const Vec4f& i_vector ) const
    {
        return AlmostEqual( m_elements[ 0 ], i_vector.m_elements[ 0 ] ) &&
               AlmostEqual( m_elements[ 1 ], i_vector.m_elements[ 1 ] ) &&
               AlmostEqual( m_elements[ 2 ], i_vector.m_elements[ 2 ] ) &&
               AlmostEqual( m_elements[ 3 ], i_vector.m_elements[ 3 ] );
    }

    /// Not equal operator
    GM_HOST_DEVICE inline bool operator!=( const Vec4f& i_vector ) const
    {
        return !( ( *this ) == i_vector );
    }

    /// Get the number of elements in this vector.
    GM_HOST_DEVICE inline static size_t GetElementSize()
    {
        return 4;
    }

    /// Are any of the element values NaNs?
    GM_HOST_DEVICE inline bool HasNans() const
    {
        return std::isnan( m_elements[ 0 ] ) || std::isnan( m_elements[ 1 ] ) || std::isnan( m_elements[ 2 ] ) ||
               std::isnan( m_elements[ 3 ] );
    }

    /// Get the string representation.  For debugging purposes.
    ///
    /// \param i_classPrefix optional string to prefix class tokens.
    ///
    /// \return descriptive string representing this type instance.
    inline std::string GetString( const std::string& i_classPrefix = std::string() ) const
    {
        std::stringstream ss;
        ss << i_classPrefix << "Vec4f( ";
        ss << m_elements[ 0 ];
        ss << ", ";
        ss << m_elements[ 1 ];
        ss << ", ";
        ss << m_elements[ 2 ];
        ss << ", ";
        ss << m_elements[ 3 ];
        ss << " )";
        return ss.str();
    }

private:
    float m_elements[ 4 ] = {0.0f, 0.0f, 0.0f, 0.0f};
};

/// Vector-scalar multiplication.
GM_HOST_DEVICE inline Vec4f operator*( const Vec4f& i_vector, const float& i_scalar )
{
    GM_ASSERT( !i_vector.HasNans() );
    return Vec4f( i_vector[ 0 ] * i_scalar,
                  i_vector[ 1 ] * i_scalar,
                  i_vector[ 2 ] * i_scalar,
                  i_vector[ 3 ] * i_scalar );
}

/// Scalar-vector multiplication.
GM_HOST_DEVICE inline Vec4f operator*( const float& i_scalar, const Vec4f& i_vector )
{
    GM_ASSERT( !i_vector.HasNans() );
    return Vec4f( i_vector[ 0 ] * i_scalar,
                  i_vector[ 1 ] * i_scalar,
                  i_vector[ 2 ] * i_scalar,
                  i_vector[ 3 ] * i_scalar );
}

/// Operator overload for << to enable writing the string representation of \p i_vector into an output
/// stream \p o_outputStream.
///
/// \param o_outputStream the output stream to write into.
/// \param i_vector the source vector value type.
///
/// \return the output stream.
inline std::ostream& operator<<( std::ostream& o_outputStream, const Vec4f& i_vector )
{
    o_outputStream << i_vector.GetString();
    return o_outputStream;
}

GM_NS_CLOSE