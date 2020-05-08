#include "valueTypes.h"

#include <stdio.h>

void MatrixArrayProduct_CPU( const Mat4f* i_matricesA, const Mat4f* i_matricesB, int i_numMatrices, Mat4f* o_matrices )
{
    for ( int matrixIndex = 0; matrixIndex < i_numMatrices; ++matrixIndex )
    {
        o_matrices[ matrixIndex ] = i_matricesA[ matrixIndex ] * i_matricesB[ matrixIndex ];
    }
}

__global__ void
MatrixArrayProduct_Naive( const Mat4f* i_matricesA, const Mat4f* i_matricesB, int i_numMatrices, Mat4f* o_matrices )
{
    int matrixIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( matrixIndex >= i_numMatrices )
    {
        return;
    }

    o_matrices[ matrixIndex ] = i_matricesA[ matrixIndex ] * i_matricesB[ matrixIndex ];
}

