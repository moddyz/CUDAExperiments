#pragma once

/// \file performance.h
///
/// A set tools for computing performance metrics in CUDA programming.

#include <cudaTools/diagnostic.h>

#include <limits>
#include <vector>

/// Compute the theoretical memory bandwidth of the current CUDA device.
/// The bandwidth proportional to memory clock rate and bandwidth.
///
/// \return bandwidth in units of gigabytes per second.
inline double CudaComputeTheoreticalMemoryBandwidth()
{
    int currentDeviceIndex;
    CUDA_CHECK( cudaGetDevice( &currentDeviceIndex ) );

    cudaDeviceProp deviceProp;
    CUDA_CHECK( cudaGetDeviceProperties( &deviceProp, currentDeviceIndex ) );

    double bytesPerSecond =
        ( ( /* KHz -> Hertz */ deviceProp.memoryClockRate * 1000.0 ) *
          ( /* Bits -> Bytes */ deviceProp.memoryBusWidth / 8.0 ) * ( /* Double data rate */ 2.0 ) );

    // Convert to GB per sec.
    return bytesPerSecond / 1e9;
}

/// Compute the effective memory bandwidth, based on the number of bytes read \p i_numBytesRead and written to \p
/// i_numBytesRead, and elapsed time \p i_msElapsed.
///
/// \return bandwidth in units of gigabytes per second.
inline double CudaComputeEffectiveMemoryBandwidth( size_t i_numBytesRead, size_t i_numBytesWritten, double i_msElapsed )
{
    // Unit conversion constants.
    constexpr double secondsPerMillisecond = 1e-3;
    constexpr double bytesPerGigabyte      = 1e9;

    // Effective bandwidth is bytes read and written divided by time.
    double bytesPerSecond =
        ( ( double ) ( i_numBytesRead + i_numBytesWritten ) ) / ( i_msElapsed * secondsPerMillisecond );

    return bytesPerSecond / bytesPerGigabyte;
}

/// Prints performance attributes specific to the current CUDA device.
inline void CudaPrintDevicePerformanceAttributes()
{
    int currentDeviceIndex;
    CUDA_CHECK( cudaGetDevice( &currentDeviceIndex ) );
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, currentDeviceIndex );
    printf( "\n[CUDA Device %d: \"%s\"]\n", currentDeviceIndex, deviceProp.name );
    printf( "   Theoretical Memory Bandwidth:           %f GB/s\n", CudaComputeTheoreticalMemoryBandwidth() );
    printf( "\n" );
}

/// \struct CudaKernelLaunchParams
///
/// Stores parameters required launch a CUDA kernel.
struct CudaKernelLaunchParams
{
    const char*        name   = nullptr; /// Name of the kernel.
    void*              kernel = nullptr; /// Pointer to the kernel function.
    std::vector<void*> args;             /// Args the arguments to the kernel function.
    dim3               grid;             /// The block launch dimensions.
    dim3               block;            /// The thread launch dimensions.
};

/// \class CudaTimer
///
/// A simple CUDA timer using cuda events.  Follows the RAII pattern.
class CudaTimer
{
public:
    /// Default constructor.
    ///
    /// The timer starts on construction.
    CudaTimer()
    {
        // Initialize events.
        CUDA_CHECK( cudaEventCreate( &m_start ) );
        CUDA_CHECK( cudaEventCreate( &m_stop ) );

        // Start timer.
        CUDA_CHECK( cudaEventRecord( m_start, 0 ) );
    }

    /// Stop the timer and return elapsed time.
    ///
    /// \return The elapsed time in milliseconds.
    inline float Stop()
    {
        // Stop timer, and get elapsed time in milliseconds.
        CUDA_CHECK( cudaEventRecord( m_stop, 0 ) );
        CUDA_CHECK( cudaEventSynchronize( m_stop ) );
        float elapsedMs;
        CUDA_CHECK( cudaEventElapsedTime( &elapsedMs, m_start, m_stop ) );
        return elapsedMs;
    }

    ~CudaTimer()
    {
        // Destroy the eve
        CUDA_CHECK( cudaEventDestroy( m_start ) );
        CUDA_CHECK( cudaEventDestroy( m_stop ) );
    }

private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};

/// Execute a kernel benchmark.
/// The kernel is timed, and the effective memory bandwidth is computed.
/// Results are printed to stdout.
///
/// \param i_kernelParams cuda kernel launch parameters.
/// \param i_numBytesRead number of bytes of memory read in the kernel, used for computing effective bandwidth.
/// \param i_numBytesWritten number of bytes of memory written in the kernel, used for computing effective bandwidth.
/// \param i_numIterations number of iterations to execute the kernel, for aggregation of timings.
inline void CudaKernelBenchmark( CudaKernelLaunchParams& i_kernelParams,
                                 size_t                  i_numBytesRead,
                                 size_t                  i_numBytesWritten,
                                 size_t                  i_numIterations = 100 )
{
    double totalElapsed = 0.0;
    double minElapsed   = std::numeric_limits<double>::max();
    double maxElapsed   = std::numeric_limits<double>::min();

    // Create events for timings.
    for ( size_t i = 0; i < i_numIterations; ++i )
    {
        CudaTimer timer;

        // Execute kernel.
        CUDA_CHECK( cudaLaunchKernel( i_kernelParams.kernel,
                                      i_kernelParams.grid,
                                      i_kernelParams.block,
                                      i_kernelParams.args.data(),
                                      0,
                                      nullptr ) );

        float elapsedMs = timer.Stop();

        // Update total, min, & max elapsed times.
        totalElapsed += elapsedMs;
        minElapsed = std::min( minElapsed, ( double ) elapsedMs );
        maxElapsed = std::max( maxElapsed, ( double ) elapsedMs );
    }

    // Compute the average.
    double averageElapsed = totalElapsed / ( double ) i_numIterations;

    // Print results.
    printf( "<<< Kernel Benchmark \"%s\" >>>\n", i_kernelParams.name );
    printf( "   Number of iterations:                   %i\n", i_numIterations );
    printf( "   Total elapsed time:                     %f ms\n", totalElapsed );
    printf( "   Average elapsed time:                   %f ms\n", averageElapsed );
    printf( "   Min elapsed time:                       %f ms\n", minElapsed );
    printf( "   Max elapsed time:                       %f ms\n", maxElapsed );
    printf( "   Average effective bandwidth:            %f GB/s\n",
            CudaComputeEffectiveMemoryBandwidth( i_numBytesRead, i_numBytesWritten, averageElapsed ) );
}
