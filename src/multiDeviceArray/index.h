#pragma once

/// \file index.h
///
/// Memory indexed access operator.

#include "falseType.h"
#include "residency.h"

#include <cuda_runtime.h>

/// \struct MemoryIndex
///
/// Template prototype for a buffer index operation.
template <Residency ResidencyT>
struct MemoryIndex
{
};

template <>
struct MemoryIndex<Host>
{
    template <typename ValueT>
    static inline ValueT& Execute( ValueT* i_buffer, size_t i_index )
    {
        return i_buffer[ i_index ];
    }
};

template <>
struct MemoryIndex<CUDA>
{
    template <typename ValueT>
    static inline ValueT& Execute( ValueT* i_buffer, size_t i_index )
    {
        static_ASSERT( FalseType<ValueT>::value, "Cannot index into cuda buffer from host code." );
        return ValueT();
    }
};
