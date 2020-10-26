// CUDA.
#include <cuda_runtime.h>

// Relative.
#include "matrixArrayProduct.h"

// Local tools.
#include <cudaTools/error.h>
#include <cudaTools/performance.h>

// GraphicsMath.
#include <gm/functions/matrixProduct.h>
#include <gm/functions/setIdentity.h>

// Thirdparty.
#include <cxxopts.hpp>

/// Helper function for setting a value \p i_matrix for \p i_arraySize elements in \p o_matrices.
void SetMatrixArrayValue( const gm::Mat4f& i_matrix, int i_arraySize, gm::Mat4f* o_matrices )
{
    for ( int matrixIndex = 0; matrixIndex < i_arraySize; ++matrixIndex )
    {
        o_matrices[ matrixIndex ] = i_matrix;
    }
}

/// Helper function for checking that all the values in the two matrices arrays are equal.
void CheckMatrixArrays( const gm::Mat4f* i_matrixA, const gm::Mat4f* i_matrixB, int i_arraySize )
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
    options.add_options()( "n,arraySize", "Size of matrix array.", cxxopts::value<int>()->default_value( "10000" ) );
    auto result    = options.parse( i_argc, i_argv );
    int  arraySize = result[ "arraySize" ].as<int>();

    // Compute amount of memory to allocate.
    size_t numBytes = arraySize * sizeof( gm::Mat4f );

    // Print current device perf attributes.
    CudaPrintDevicePerformanceAttributes();

    // Allocate host memory.
    gm::Mat4f* matricesA   = ( gm::Mat4f* ) malloc( numBytes );
    gm::Mat4f* matricesB   = ( gm::Mat4f* ) malloc( numBytes );
    gm::Mat4f* matricesC   = ( gm::Mat4f* ) malloc( numBytes );
    gm::Mat4f* matricesRef = ( gm::Mat4f* ) malloc( numBytes );

    // Set host values.
    gm::Mat4f identity;
    gm::SetIdentity( identity );
    SetMatrixArrayValue( identity, arraySize, matricesA );
    SetMatrixArrayValue( identity, arraySize, matricesB );

    // Compute CPU output.
    MatrixArrayProduct_CPU( matricesA, matricesB, arraySize, matricesRef );

    // Allocate device memory.
    gm::Mat4f* matricesADevice;
    gm::Mat4f* matricesBDevice;
    gm::Mat4f* matricesCDevice;
    CUDA_CHECK( cudaMalloc( ( void** ) &matricesADevice, numBytes ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &matricesBDevice, numBytes ) );
    CUDA_CHECK( cudaMalloc( ( void** ) &matricesCDevice, numBytes ) );

    // Upload host memory -> device.
    CUDA_CHECK( cudaMemcpy( /*dst*/ matricesADevice, /*src*/ matricesA, numBytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK( cudaMemcpy( /*dst*/ matricesBDevice, /*src*/ matricesB, numBytes, cudaMemcpyHostToDevice ) );

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
    CUDA_CHECK( cudaMemcpy( /*dst*/ matricesC, /*src*/ matricesCDevice, numBytes, cudaMemcpyDeviceToHost ) );
    CheckMatrixArrays( matricesC, matricesRef, arraySize );

    // Free allocated resources
    free( matricesA );
    free( matricesB );
    free( matricesC );
    free( matricesRef );
    CUDA_CHECK( cudaFree( matricesADevice ) );
    CUDA_CHECK( cudaFree( matricesBDevice ) );
    CUDA_CHECK( cudaFree( matricesCDevice ) );
}
