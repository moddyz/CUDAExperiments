// CUDA.
#include <cuda_runtime.h>

// Relative.
#include "matrixArrayProduct.h"

// Local tools.
#include <cudaExperiments/error.h>
#include <cudaExperiments/math.h>
#include <cudaExperiments/performance.h>
#include <cudaExperiments/valueTypes.h>

// Thirdparty.
#include <cxxopts.hpp>

/// Helper function for setting a value \p i_matrix for \p i_arraySize elements in \p o_matrices.
void SetMatrixArrayValue( const Mat4f& i_matrix, int i_arraySize, Mat4f* o_matrices )
{
    for ( int matrixIndex = 0; matrixIndex < i_arraySize; ++matrixIndex )
    {
        o_matrices[ matrixIndex ] = i_matrix;
    }
}

/// Helper function for checking that all the values in the two matrices arrays are equal.
void CheckMatrixArrays( const Mat4f* i_matrixA, const Mat4f* i_matrixB, int i_arraySize )
{
    for ( int matrixIndex = 0; matrixIndex < i_arraySize; ++matrixIndex )
    {
        if ( i_matrixA[ matrixIndex ] != i_matrixB[ matrixIndex ] )
        {
            fprintf( stderr,
                     "MatrixA[ %d ] != MatrixB[ %d ],\n%s != %s\n",
                     matrixIndex,
                     matrixIndex,
                     i_matrixA[ matrixIndex ].GetString().c_str(),
                     i_matrixB[ matrixIndex ].GetString().c_str() );
            return;
        }
    }
}

int main( int i_argc, char** i_argv )
{
    // Parse command line arguments.
    cxxopts::Options options( "cudaMatrixArrayOps",
                              "Testing various implementations of 4 by 4 matrix array operations." );
    options.add_options()( "n,arraySize", "Size of matrix array.", cxxopts::value< int >()->default_value( "10000" ) );
    auto result    = options.parse( i_argc, i_argv );
    int  arraySize = result[ "arraySize" ].as< int >();

    // Compute amount of memory to allocate.
    size_t numBytes = arraySize * sizeof( Mat4f );

    // Print current device perf attributes.
    CudaPrintDevicePerformanceAttributes();

    // Allocate host memory.
    Mat4f* matricesA   = ( Mat4f* ) malloc( numBytes );
    Mat4f* matricesB   = ( Mat4f* ) malloc( numBytes );
    Mat4f* matricesC   = ( Mat4f* ) malloc( numBytes );
    Mat4f* matricesRef = ( Mat4f* ) malloc( numBytes );

    // Set host values.
    SetMatrixArrayValue( Mat4f::Identity(), arraySize, matricesA );
    SetMatrixArrayValue( Mat4f::Identity(), arraySize, matricesB );

    // Compute CPU output.
    MatrixArrayProduct_CPU( matricesA, matricesB, arraySize, matricesRef );

    // Allocate device memory.
    Mat4f* matricesADevice;
    Mat4f* matricesBDevice;
    Mat4f* matricesCDevice;
    CUDA_CHECK_ERROR_FATAL( cudaMalloc( ( void** ) &matricesADevice, numBytes ) );
    CUDA_CHECK_ERROR_FATAL( cudaMalloc( ( void** ) &matricesBDevice, numBytes ) );
    CUDA_CHECK_ERROR_FATAL( cudaMalloc( ( void** ) &matricesCDevice, numBytes ) );

    // Upload host memory -> device.
    CUDA_CHECK_ERROR_FATAL(
        cudaMemcpy( /*dst*/ matricesADevice, /*src*/ matricesA, numBytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK_ERROR_FATAL(
        cudaMemcpy( /*dst*/ matricesBDevice, /*src*/ matricesB, numBytes, cudaMemcpyHostToDevice ) );

    // Compute grid & block size based on array size.
    {
        CudaKernelLaunchParams params;
        params.name    = "MatrixArrayProduct_Naive";
        params.kernel  = ( void* ) MatrixArrayProduct_Naive;
        params.block.x = 256;
        params.grid.x  = ( arraySize + params.block.x - 1 ) / params.block.x;
        params.args    = {&matricesADevice, &matricesBDevice, &arraySize, &matricesCDevice};
        CudaKernelBenchmark( params, /* bytesRead */ numBytes * 2, /* bytesWritten */ numBytes );
    }

    // Download computed matrices, and verify against CPU results.
    CUDA_CHECK_ERROR_FATAL(
        cudaMemcpy( /*dst*/ matricesC, /*src*/ matricesCDevice, numBytes, cudaMemcpyDeviceToHost ) );
    CheckMatrixArrays( matricesC, matricesRef, arraySize );
}
