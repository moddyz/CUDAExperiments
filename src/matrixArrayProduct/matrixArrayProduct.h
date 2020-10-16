#pragma once

/// \file matrixArrayOps/matrixArrayProduct.h
///
/// Matrix array product (MAP)
///
/// Element-wise multiplication of two arrays of 4 x 4 matrices (commonly used in computer graphics).

namespace gm
{
class Mat4f;
}

/// CPU implementation of a MAP.
void MatrixArrayProduct_CPU( const gm::Mat4f* i_matricesA,
                             const gm::Mat4f* i_matricesB,
                             int              i_numMatrices,
                             gm::Mat4f*       o_matrices );

/// Naive CUDA implementation of MAP.
__global__ void MatrixArrayProduct_Naive( const gm::Mat4f* i_matricesA,
                                          const gm::Mat4f* i_matricesB,
                                          int              i_numMatrices,
                                          gm::Mat4f*       o_matrices );
