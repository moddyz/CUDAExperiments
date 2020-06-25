#include <cstdio>
#include <stdlib.h>
#include <string>

#include <cuda_runtime.h>

#include <cubase/error.h>

/// Utility program for querying available CUDA devices, and their respective properties.

static const char* GetEnabledToken( bool i_status )
{
    return i_status ? "Enabled" : "Disabled";
}

int main()
{
    printf( "[cudaQueryDeviceProperties]\n" );

    int deviceCount = 0;
    CUDA_ERROR_FATAL( cudaGetDeviceCount( &deviceCount ) );

    // Iterate over each device and query their properties.
    for ( int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex )
    {
        cudaSetDevice( deviceIndex );
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties( &deviceProp, deviceIndex );

        printf( "\nDevice %d: \"%s\"\n\n", deviceIndex, deviceProp.name );

        //
        // Overview
        //

        printf( "  Overview\n" );

        // The latest version of CUDA supported by the driver.
        int driverVersion = 0;
        cudaDriverGetVersion( &driverVersion );

        // The version of the CUDA libraries linked into this runtime.
        int runtimeVersion = 0;
        cudaRuntimeGetVersion( &runtimeVersion );
        printf( "    CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
                driverVersion / 1000,
                ( driverVersion % 100 ) / 10,
                runtimeVersion / 1000,
                ( runtimeVersion % 100 ) / 10 );

        // CUDA Device Capability.
        printf( "    CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor );
        printf( "\n" );

        //
        // Memory
        //

        printf( "  Memory\n" );

        printf( "    Total amount of global memory:                 %.0f MBytes\n",
                static_cast< float >( deviceProp.totalGlobalMem / ( /* MBytes -> Bytes */ 1024.0f * 1024.0f ) ) );
        printf( "    Total amount of constant memory:               %.0f Bytes\n",
                static_cast< float >( deviceProp.totalConstMem ) );
        printf( "    Memory Bus Width:                              %.0f Bytes\n",
                static_cast< float >( deviceProp.memoryBusWidth / 8 /* Bits -> Bytes */ ) );
        printf( "    Memory Clock Rate:                             %.0f KHz\n",
                static_cast< float >( deviceProp.memoryClockRate ) );

        printf( "\n" );

        //
        // Device features (properties which don't really fit in other categories)
        //

        printf( "  Device Features\n" );

        // ECC == Error-Correcting code memory.  It is a type of storage which can detect and correct most
        // common kinds of internal data corruption.
        printf( "    ECC Support:                                   %s\n", GetEnabledToken( deviceProp.ECCEnabled ) );

        // Asynchronous engine count.  I think this refers to the # of copy engines.
        //
        // For devices with 2 copy engines - which means two concurrent copy operations can overlap:
        // - Host -> Device (upload)
        // - Device -> Host (download)
        //
        // For devices with 3 copy engines - which means _three_ concurrent copy operations can overlap, with
        // the third operation being a:
        // - Device -> Device via NVLINK
        printf( "    Asynchronous Engine Count:                     %d\n", deviceProp.asyncEngineCount );
    }

    return EXIT_SUCCESS;
}
