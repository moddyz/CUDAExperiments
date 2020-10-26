#include <cassert>
#include <stdio.h>

#include <cudaTools/error.h>
#include <cudaTools/performance.h>

// Fixed number of threads per block.
static const int THREADS_PER_BLOCK = 32;

// Simple kernel which adds 1 to every array entry.
__global__ void AddOne( int i_numElements, float* o_array )
{
    int globalIndex = threadIdx.x + ( blockIdx.x * blockDim.x );
    if ( globalIndex > i_numElements )
    {
        return;
    }

    o_array[ globalIndex ] += 1.0f;
}

// Simple kernel which multiplies each every by 5.
__global__ void MultiplyFive( int i_numElements, float* o_array )
{
    int globalIndex = threadIdx.x + ( blockIdx.x * blockDim.x );
    if ( globalIndex > i_numElements )
    {
        return;
    }

    o_array[ globalIndex ] *= 5.0f;
}

int main( int i_argc, char** i_argv )
{
    // Allocate cuda stream.
    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );

    // Allocate host array and fill with "1"s.
    int    numElements = 128;
    float* hostArray   = ( float* ) malloc( sizeof( float ) * numElements );
    for ( size_t i = 0; i < numElements; ++i )
    {
        hostArray[ i ] = 1.0f;
    }

    // Allocate device array.
    int    numBlocks   = ( numElements + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    float* deviceArray = nullptr;
    CUDA_CHECK( cudaMalloc( &deviceArray, sizeof( float ) * numElements ) );

    {
        CudaTimer timer;
        // Upload asynchronously with stream.
        CUDA_CHECK(
            cudaMemcpyAsync( deviceArray, hostArray, sizeof( float ) * numElements, cudaMemcpyHostToDevice, stream ) );

        // Execute kernels with stream & synchronize stream.
        AddOne<<< numBlocks, THREADS_PER_BLOCK, 0, stream >>>( numElements, deviceArray );
        MultiplyFive<<< numBlocks, THREADS_PER_BLOCK, 0, stream >>>( numElements, deviceArray );

        // Download device -> host.
        CUDA_CHECK(
            cudaMemcpyAsync( hostArray, deviceArray, sizeof( float ) * numElements, cudaMemcpyDeviceToHost, stream ) );
        cudaStreamSynchronize( stream );
        printf( "Took: %f ms\n", timer.Stop() );
    }

    for ( size_t i = 0; i < numElements; ++i )
    {
        assert( hostArray[ i ] == 10.0f );
    }

    // Deallocate resources.
    free( hostArray );
    CUDA_CHECK( cudaFree( deviceArray ) );
    CUDA_CHECK( cudaStreamDestroy( stream ) );
}
