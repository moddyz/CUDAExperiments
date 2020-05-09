#pragma once

/// \file performance.h
///
/// A set tools for computing performance metrics in CUDA programming.

#include <cudaExperiments/error.h>

/// Compute the theoretical memory bandwidth of the current CUDA device.
/// The bandwidth proportional to memory clock rate and bandwidth.
///
/// \return bandwidth in units of gigabytes per second.
inline double CudaComputeTheoreticalMemoryBandwidth()
{
    int currentDeviceIndex;
    CUDA_CHECK_ERROR_FATAL( cudaGetDevice( &currentDeviceIndex ) );

    cudaDeviceProp deviceProp;
    CUDA_CHECK_ERROR_FATAL( cudaGetDeviceProperties( &deviceProp, currentDeviceIndex ) );

    double bytesPerSecond =
        ( ( /* KHz -> Hertz */ deviceProp.memoryClockRate * 1000.0 ) *
          ( /* Bits -> Bytes */ deviceProp.memoryBusWidth / 8.0 ) * ( /* Double data rate */ 2.0 ) );

    // Convert to GB per sec.
    return bytesPerSecond / 1e9;
}

/// Compute the effective memory bandwidth, based on the number of bytes read \p i_numBytesRead and written to \p
/// i_numBytesRead, and elapsed kernel time \p i_msElapsed.
///
/// \return bandwidth in units of gigabytes per second.
inline double CudaComputeEffectiveMemoryBandwidth( size_t i_numBytesRead, size_t i_numBytesWritten, double i_msElapsed )
{
    double bytesPerSecond =
        ( ( double ) ( i_numBytesRead + i_numBytesWritten ) ) / ( /* Milliseconds -> Seconds */ i_msElapsed * 1e-3 );
    return bytesPerSecond / 1e9;
}


