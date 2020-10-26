#include <stdio.h>

#include <cudaTools/error.h>

__global__ void MyKernel( size_t i_pitch, int i_numCols, int i_numRows, float* o_array )
{
    for ( int rowIndex = 0; rowIndex < i_numRows; ++rowIndex )
    {
        // Offset into the row at `rowIndex`.
        float* rowData = ( float* ) ( ( char* ) o_array + rowIndex * i_pitch );
        for ( int colIndex = 0; colIndex < i_numCols; ++colIndex )
        {
            rowData[ colIndex ] = 0.0f;
        }
    }
}

int main( int i_argc, char** i_argv )
{
    int    numCols = 64;
    int    numRows = 64;
    size_t pitch   = 0;
    float* array   = nullptr;

    cudaMallocPitch( &array,
                     &pitch,                    // Number of bytes allocated for a single row (includes padding)
                     numCols * sizeof( float ), // The width (in bytes) of each row.
                     numRows );

    MyKernel<<< 1, 1 >>>( pitch, numCols, numRows, array );
    cudaDeviceSynchronize();
    
    // Free allocated resources.
    CUDA_CHECK( cudaFree( array ) );
}
