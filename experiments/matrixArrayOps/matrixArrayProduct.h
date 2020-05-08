#pragma once

/// \page MatrixArrayProduct Matrix Array Product
///
/// Element-wise multiplication of two arrays of 4 x 4 matrices (commonly used in computer graphics).

#include "valueTypes.h"

__global__ void
MatrixArrayProduct_Naive( const Mat4f* i_matricesA, const Mat4f* i_matricesB, int i_numMatrices, Mat4f* o_matrices );
