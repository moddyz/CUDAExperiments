#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/// \file cudaUtils.h
///
/// A set of useful utilities for CUDA programming.

/// \macro CUDA_CHECK_ERROR_CONTINUE
///
/// Check the error status.  On non-success, log to stderr and continue exceution.
#define CUDA_CHECK_ERROR_CONTINUE( val )                                                                               \
    CudaCheckError< CudaErrorSeverity::Continue >( ( val ), #val, __FILE__, __LINE__ )

/// \macro CUDA_CHECK_ERROR_FATAL
///
/// Check the error status.  On non-success, log to stderr and exit the program.
#define CUDA_CHECK_ERROR_FATAL( val ) CudaCheckError< CudaErrorSeverity::Fatal >( ( val ), #val, __FILE__, __LINE__ )

/// \enum CudaErrorSeverity
///
/// Severity of CUDA error.
enum class CudaErrorSeverity : char
{
    Continue = 0,
    Fatal = 1
};

/// Not intended to be used directly - use \ref CUDA_CHECK_ERROR_FATAL and \ref CUDA_CHECK_ERROR_CONTINUE instead.
template < CudaErrorSeverity ErrorSeverityValue >
void CudaCheckError( cudaError_t i_error, const char* i_function, const char* i_file, int i_line )
{
    if ( i_error != cudaSuccess )
    {
        fprintf( stderr,
                 "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                 i_file,
                 i_line,
                 static_cast< unsigned int >( i_error ),
                 cudaGetErrorName( i_error ),
                 i_function );

        if constexpr ( ErrorSeverityValue == CudaErrorSeverity::Fatal )
        {
            exit( EXIT_FAILURE );
        }
    }
}

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

