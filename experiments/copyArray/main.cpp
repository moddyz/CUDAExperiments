// CUDA.
#include <cuda_runtime.h>

// Relative.
#include "copyArray.h"

// Local tools.
#include <cudaExperiments/error.h>
#include <cudaExperiments/math.h>
#include <cudaExperiments/performance.h>

// Thirdparty.
#include <cxxopts.hpp>

int main( int i_argc, char** i_argv )
{
    // Parse command line arguments.
    cxxopts::Options options( "cudaCopyArray", "Baseline test for a simple array copy kernel." );
    options.add_options()( "n,arraySize", "Size of array.", cxxopts::value< int >()->default_value( "10000" ) );
    auto result    = options.parse( i_argc, i_argv );
    int  arraySize = result[ "arraySize" ].as< int >();

    // Compute amount of memory to allocate.
    size_t numBytes = arraySize * sizeof( float );

    // Print current device perf attributes.
    CudaPrintDevicePerformanceAttributes();

    // Allocate host memory.
    float* arrayA = ( float* ) malloc( numBytes );
    float* arrayB = ( float* ) malloc( numBytes );

    // Allocate device memory.
    float* arrayADevice;
    float* arrayBDevice;
    CUDA_CHECK_ERROR_FATAL( cudaMalloc( ( void** ) &arrayADevice, numBytes ) );
    CUDA_CHECK_ERROR_FATAL( cudaMalloc( ( void** ) &arrayBDevice, numBytes ) );

    // Upload host memory -> device.
    CUDA_CHECK_ERROR_FATAL( cudaMemcpy( /*dst*/ arrayADevice, /*src*/ arrayA, numBytes, cudaMemcpyHostToDevice ) );
    CUDA_CHECK_ERROR_FATAL( cudaMemcpy( /*dst*/ arrayBDevice, /*src*/ arrayB, numBytes, cudaMemcpyHostToDevice ) );

    // Compute grid & block size based on array size.
    {
        CudaKernelLaunchParams params;
        params.name    = "CopyArray";
        params.kernel  = ( void* ) CopyArray< float >;
        params.block.x = 256;
        params.grid.x  = ( arraySize + params.block.x - 1 ) / params.block.x;
        params.args    = {&arraySize, &arrayADevice, &arrayBDevice};
        CudaKernelBenchmark( params, /* bytesRead */ numBytes * 2, /* bytesWritten */ numBytes );
    }

    // Download computed matrices, and verify against CPU results.
    CUDA_CHECK_ERROR_FATAL( cudaMemcpy( /*dst*/ arrayB, /*src*/ arrayBDevice, numBytes, cudaMemcpyDeviceToHost ) );
}
