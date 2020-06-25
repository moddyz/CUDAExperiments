#include <gm/functions/matrixProduct.h>

#include <stdio.h>

void MatrixArrayProduct_CPU( const gm::Mat4f* i_matricesA,
                             const gm::Mat4f* i_matricesB,
                             int              i_numMatrices,
                             gm::Mat4f*       o_matrices )
{
    for ( int matrixIndex = 0; matrixIndex < i_numMatrices; ++matrixIndex )
    {
        o_matrices[ matrixIndex ] = gm::MatrixProduct( i_matricesA[ matrixIndex ], i_matricesB[ matrixIndex ] );
    }
}

__global__ void MatrixArrayProduct_Naive( const gm::Mat4f* i_matricesA,
                                          const gm::Mat4f* i_matricesB,
                                          int              i_numMatrices,
                                          gm::Mat4f*       o_matrices )
{
    int matrixIndex = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    if ( matrixIndex >= i_numMatrices )
    {
        return;
    }

    o_matrices[ matrixIndex ] = gm::MatrixProduct( i_matricesA[ matrixIndex ], i_matricesB[ matrixIndex ] );
}
