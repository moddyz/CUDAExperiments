// A simple example of passing a device function pointer into a kernel for execution.

#include <cassert>
#include <cudaTools/diagnostic.h>
#include <stdio.h>

// Fixed number of threads per block.
static const int THREADS_PER_BLOCK = 32;

// Function pointer signature type.
typedef void ( *OperationFn )( int, float* );

// Device function definition
__device__ void Add( int i_index, float* o_array )
{
    o_array[ i_index ] += 1.0f;
}

// Store the function pointer of the Add operation.
__device__ OperationFn g_deviceFn = Add;

// Dispatching kernel.
__global__ void Dispatch( OperationFn i_operation, int i_numElements, float* o_array )
{
    int globalIndex = threadIdx.x + ( blockIdx.x * blockDim.x );
    if ( globalIndex > i_numElements )
    {
        return;
    }

    i_operation( globalIndex, o_array );
}

int main( int i_argc, char** i_argv )
{
    // Allocate host array and fill with "1"s.
    int    numElements = 128;
    float* hostArray   = ( float* ) malloc( sizeof( float ) * numElements );
    for ( size_t i = 0; i < numElements; ++i )
    {
        hostArray[ i ] = 1.0f;
    }

    // Allocate device array and upload data.
    int    numBlocks   = ( numElements + THREADS_PER_BLOCK - 1 ) / THREADS_PER_BLOCK;
    float* deviceArray = nullptr;
    CUDA_CHECK( cudaMalloc( &deviceArray, sizeof( float ) * numElements ) );
    CUDA_CHECK( cudaMemcpy( deviceArray, hostArray, sizeof( float ) * numElements, cudaMemcpyHostToDevice ) );

    // Copy the value of the device function pointer into host.
    // The device function pointer _cannot_ be accessed directly from host code!
    OperationFn hostFn = nullptr;
    CUDA_CHECK( cudaMemcpyFromSymbol( &hostFn, g_deviceFn, sizeof( OperationFn ) ) );

    // Execute kernel, passing in the device function address stored in our hostFn variable.
    Dispatch<<<numBlocks, THREADS_PER_BLOCK>>>( hostFn, numElements, deviceArray );

    // Download device -> host.
    CUDA_CHECK( cudaMemcpy( hostArray, deviceArray, sizeof( float ) * numElements, cudaMemcpyDeviceToHost ) );

    for ( size_t i = 0; i < numElements; ++i )
    {
        assert( hostArray[ i ] == 2.0f );
    }

    // Deallocate resources.
    free( hostArray );
    CUDA_CHECK( cudaFree( deviceArray ) );
}
