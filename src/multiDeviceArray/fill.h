#pragma once

/// \file memory/fill.h
///
/// Memory fill operation.

#include "allocate.h"
#include "copy.h"
#include "residency.h"

#include <cuda_runtime.h>

/// \struct MemoryFill
///
/// Template prototype for a fill operation.
template <Residency ResidencyT>
struct MemoryFill
{
};

template <>
struct MemoryFill<Host>
{
    template <typename ValueT>
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_dstBuffer )
    {
        ASSERT( o_dstBuffer != nullptr );

        for ( size_t elementIndex = 0; elementIndex < i_numElements; ++elementIndex )
        {
            o_dstBuffer[ elementIndex ] = i_value;
        }

        return true;
    }
};

template <>
struct MemoryFill<CUDA>
{
    template <typename ValueT>
    static inline bool Execute( size_t i_numElements, const ValueT& i_value, ValueT* o_dstBuffer )
    {
        // MemoryAllocate and fill on the host.
        size_t numBytes  = i_numElements * sizeof( ValueT );
        void*  srcBuffer = MemoryAllocate<Host>::Execute( numBytes );
        if ( srcBuffer == nullptr )
        {
            return false;
        }

        ASSERT( MemoryFill<Host>::Execute( i_numElements, i_value, static_cast<ValueT*>( srcBuffer ) ) );

        // Then copy to GPU.
        bool result = MemoryCopy<Host, CUDA>::Execute( numBytes, srcBuffer, o_dstBuffer );

        // Delete temporary buffer.
        ASSERT( MemoryDeallocate<Host>::Execute( srcBuffer ) );

        return result;
    }
};
