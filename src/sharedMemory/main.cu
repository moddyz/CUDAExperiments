#include <stdio.h>

#include <cudaTools/error.h>

static const int THREADS_PER_BLOCK = 32;

__global__ void SharedMemoryKernel( int i_numElements, float* o_array )
{
    int globalIndex = threadIdx.x + ( blockIdx.x * blockDim.x );

    // Shared block of memory PER BLOCK (of threads).
    __shared__ float sharedArray[ THREADS_PER_BLOCK ];
    sharedArray[ threadIdx.x ] = o_array[ globalIndex ];
    __syncthreads();

    for ( size_t i = 0; i < THREADS_PER_BLOCK; ++i )
    {
        o_array[ globalIndex ] += sharedArray[ i ];
    }
}

int main( int i_argc, char** i_argv )
{
    // Check shared memory capacity per block.
    int deviceCount = 0;
    CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );
    for ( int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex )
    {
        cudaSetDevice( deviceIndex );
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, deviceIndex );

        printf( "\nDevice %d: \"%s\"\n", deviceIndex, deviceProp.name );
        printf( "    Shared memory per thread block:                 %.0f MBytes\n",
                static_cast< float >( deviceProp.sharedMemPerBlock / 1024.0f ) );
        printf( "\n" );
    }

    int numElements = 128;

    // Allocate host array and fill with "1"s.
    float* hostArray = ( float* ) malloc( sizeof( float ) * numElements );
    for ( size_t i = 0; i < numElements; ++i )
    {
        hostArray[ i ] = 1.0f;
    }

    // Allocate device array and upload host -> device.
    int    numBlocks   = ( numElements + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    float* deviceArray = nullptr;
    CUDA_CHECK( cudaMalloc( &deviceArray, sizeof( float ) * numElements ) );
    CUDA_CHECK( cudaMemcpy( deviceArray, hostArray, sizeof( float ) * numElements, cudaMemcpyHostToDevice ) );

    // Execute kernel & synchronize device.
    SharedMemoryKernel<<< numBlocks, THREADS_PER_BLOCK, /* sharedMemInBytes */ THREADS_PER_BLOCK >>>( numElements, deviceArray );
    cudaDeviceSynchronize();

    // Download device -> host.
    CUDA_CHECK( cudaMemcpy( hostArray, deviceArray, sizeof( float ) * numElements, cudaMemcpyDeviceToHost ) );
    for ( size_t i = 0; i < numElements; ++i )
    {
        printf( "hostArray[ %i ] = %f\n", i, hostArray[ i ] );
    }

    // Free memory
    free( hostArray );
    cudaFree( deviceArray );
}
