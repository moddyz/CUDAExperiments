#include "copyArray.h"

template <>
__global__ void CopyArray( size_t i_size, const float* i_src, float* o_dst )
{
    int arrayIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( arrayIndex >= i_size )
    {
        return;
    }

    o_dst[ arrayIndex ] = i_src[ arrayIndex ];
}

