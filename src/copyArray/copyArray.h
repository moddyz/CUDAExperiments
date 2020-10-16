#pragma once

/// \file copyArray.h
///
/// Simple array copy.

/// Copy \p i_size elements from from \p i_src to \p o_dst;
template < typename ValueT >
__global__ void CopyArray( size_t i_size, const ValueT* i_src, ValueT* o_dst );
