/// Utility program for querying all CUDA devices, and their respective properties.

#include <cstdio>
#include <stdlib.h>

#include <cuda_runtime.h>

int main()
{
    printf( "[queryDeviceProperties]\n\n" );

    int         deviceCount = 0;
    cudaError_t cudaErr     = cudaGetDeviceCount( &deviceCount );
    if ( cudaErr != cudaSuccess )
    {
        printf( "cudaGetDeviceCount returned %d\n-> %s\n",
                static_cast< int >( cudaErr ),
                cudaGetErrorString( cudaErr ) );
        printf( "Result = FAIL\n" );
        exit( EXIT_FAILURE );
    }

    for ( int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex )
    {
        cudaSetDevice( deviceIndex );
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, deviceIndex );

        printf( "\nDevice %d: \"%s\"\n", deviceIndex, deviceProp.name );

        int driverVersion  = 0;
        int runtimeVersion = 0;
        cudaDriverGetVersion( &driverVersion );
        cudaRuntimeGetVersion( &runtimeVersion );
        printf( "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
                driverVersion / 1000,
                ( driverVersion % 100 ) / 10,
                runtimeVersion / 1000,
                ( runtimeVersion % 100 ) / 10 );
        printf( "  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor );
    }

    return 0;
}
