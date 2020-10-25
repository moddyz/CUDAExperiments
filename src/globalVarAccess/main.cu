#include <stdio.h>
#include <cassert>

#include <cudaTools/error.h>

__device__ float deviceValue;

int main()
{
    // Upload value -> deviceValue.
    float value = 3.14f;
    CUDA_CHECK( cudaMemcpyToSymbol( deviceValue, &value, sizeof( float ) ) );

    // Check deviceValue size.
    size_t deviceValueSize;
    CUDA_CHECK( cudaGetSymbolSize( &deviceValueSize, deviceValue ) );
    assert( deviceValueSize == 4 );
    printf( "deviceValue size: %lu\n", deviceValueSize );

    // Download deviceValue -> downloadedValue;
    float downloadedValue;
    CUDA_CHECK( cudaMemcpyFromSymbol( &downloadedValue, deviceValue, sizeof( float ) ) );
    printf( "Downloaded value: %f\n", downloadedValue );
}
