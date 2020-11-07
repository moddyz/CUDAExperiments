#pragma once

#include "residency.h"

#include <cudaTools/diagnostic.h>

#include <cstring>
#include <cuda_runtime.h>

/// \struct MemoryAllocate
///
/// Template prototype for a memory allocation operation.
template <Residency ResidencyT>
struct MemoryAllocate
{
};

template <>
struct MemoryAllocate<Host>
{
    static inline void* Execute( size_t i_numBytes )
    {
        ASSERT( i_numBytes != 0 );
        return malloc( i_numBytes );
    }
};

template <>
struct MemoryAllocate<CUDA>
{
    static inline void* Execute( size_t i_numBytes )
    {
        ASSERT( i_numBytes != 0 );
        void* devicePtr = nullptr;
        if ( CUDA_CHECK( cudaMallocManaged( &devicePtr, i_numBytes ) ) )
        {
            return devicePtr;
        }
        else
        {
            return nullptr;
        }
    }
};
