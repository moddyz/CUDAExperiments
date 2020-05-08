#pragma once

/// \file mathUtils.h
///
/// A set of useful math utilities.

#include <cmath>

/// Function for comparing if the difference between two values is within a small threshold.
///
/// \tparam ValueT type of the scalar values.
template < typename ValueT >
bool AlmostEqual( const ValueT& i_valueA, const ValueT& i_valueB, const ValueT& i_threshold = 0.0001 )
{
    return std::abs( i_valueA - i_valueB ) < i_threshold;
}
