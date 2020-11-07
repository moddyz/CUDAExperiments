#include <stdio.h>

#include <cudaTools/diagnostic.h>

int main( int i_argc, char** i_argv )
{
    // Create a cuda stream.
    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    
    // Get device count. 
    int deviceCount = 0;
    CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );

    for ( int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex )
    {
        // Get cuda device properties.
        cudaDeviceProp prop;
        cudaGetDeviceProperties( &prop, deviceIndex );
        
        // Query total capacity of L2 cache.
        printf("l2CacheSize: %f MB\n", (float)prop.l2CacheSize / 1024.0f );
        
        // Set persisting cache size.
        size_t size = min( int( prop.l2CacheSize * 0.75 ), prop.persistingL2CacheMaxSize );
        cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size );

        // TODO
    }
}
