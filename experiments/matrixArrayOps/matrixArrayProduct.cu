#include "valueTypes.h"

#include <stdio.h>

__global__ void
MatrixArrayProduct_Naive( const Mat4f* i_matricesA, const Mat4f* i_matricesB, int i_numMatrices, Mat4f* o_matrices )
{
    int matrixIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( matrixIndex >= i_numMatrices )
    {
        return;
    }

    // TODO.
    o_matrices[ matrixIndex ] = i_matricesA[ matrixIndex ];
}

