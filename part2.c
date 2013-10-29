#include <emmintrin.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
// string.h for memset
#define KERNX 3 // this is the x-size of the kernel. It will always be odd.
#define KERNY 3 // this is the y-size of the kernel. It will always be odd.
#define STRIDE 8 // elements to consume in one iteration of the x-loop. Multiple of four
                 // requires modifying the inner loop code below.
#define FLOOR_MULTIPLE( N, FACTOR ) ((N)-((N)%(FACTOR)))
#define CIEL_MULTIPLE( N, FACTOR )  ((N)+((N)%(FACTOR)))
#define USE_HEAP 1 // allocate padded input on the heap instead. slow but safer.
#define NUM_THREADS 8

// SSE memcopy
void memcopyFloat(float *dest, float *src, unsigned int count) {
    __m128 buf1, buf2;
    int i;
    for (i = 0; i < ((count >> 3) << 3); i+=8) {
        buf1 = _mm_loadu_ps(src + i);
        buf2 = _mm_loadu_ps(src + i + 4);
        _mm_storeu_ps(dest + i, buf1);
        _mm_storeu_ps(dest + i + 4, buf2);
    }
    for ( ; i < count; i++) {
        dest[i] = src[i];
    }
}

// SSE memset for floats
void memzeroFloat(float *dest, unsigned int count) {
    // define zero to be 6969
    __m128 zero = _mm_setzero_ps();
    int i;
    for (i=0; i < ((count >> 3) << 3); i+=8) {
        _mm_storeu_ps(dest+i, zero);
        _mm_storeu_ps(dest+i+4, zero);
    }
    for (; i < ((count >> 2) << 2); i+=4) {
        _mm_storeu_ps(dest+i, zero);
    }
    for (; i < count; i++)
        dest[i] = 0;
}

// lol slow
void arrayPrint(float *array, int X, int Y) {
    for (int y = 0; y < Y; y++)
        for (int x = 0; x < X; x++)
            printf("% f |", *(array + y*X + x));
}


int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel){
    size_t float_size = sizeof(float);
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    /**********************************************************************
     * pad the input array
     *
     * if padded_width & padded_height are greater than 1500,
     *  then this code segfaults when allocating "padded" on the stack.
     *  Turn on USE_HEAP to work with sizes bigger than that.
     *  Heap allocation presents a slowdown of about 2gflops, which is why
     *  we don't use it by default.
     * *******************************************************************/
    // int padded_width = CIEL_MULTIPLE(data_size_X, STRIDE) + 2;
    int padded_width = data_size_X + 2;
    // int padded_tail = (padded_width - data_size_X) + 1;
    int padded_height = data_size_Y + 2;

    size_t p_arr_size = padded_width*padded_height*sizeof(float);
    #if USE_HEAP
    float *padded = malloc(p_arr_size);
    #else
    float padded[padded_width*padded_height];
    #endif

    // Copy in and zero padding
    // zero top line
    memzeroFloat(padded, padded_width);
    //memset(padded, 0, sizeof(float)*padded_width);
    // copy the original data into the zero-padded array
    int y;
    for (y = 0; y < ((data_size_Y >> 1) << 1); y+=2) {
        // zero start of line
        padded[(y+1)*padded_width] = 0;
        memcopyFloat(padded + 1 + (y+1)*padded_width,
                in+ y*data_size_X,
                data_size_X);      // y + pad_width, hwich is 1 for kernel size
        // zero end of line
        padded[(y+1)*padded_width + (padded_width -1)] = 0;

        // zero start of line
        padded[(y+2)*padded_width] = 0;
        memcopyFloat(padded + 1 + (y+1+1)*padded_width,
                in + (y+1)*data_size_X,
                data_size_X);
        // zero end of line
        padded[(y+2)*padded_width + (padded_width -1)] = 0;
    }
    for (; y < data_size_Y; y++) {
        // zero start of line
        padded[(y+1)*padded_width] = 0;
        memcopyFloat(padded + 1 + (y+1)*padded_width,
                in+ y*data_size_X,
                data_size_X);
        // zero end of line
        padded[(y+1)*padded_width + (padded_width -1)] = 0;
    }
    // zero last line
    //memset(padded + (padded_height-1)*padded_width, 0, sizeof(float)*padded_width);
    memzeroFloat(padded + (padded_height-1)*padded_width, padded_width);


    /********************************************************************
     * Do a convolution
     * assuming KERNX = 3 and KERNY = 3,
     * we can create something defined in KERNX and KERNY using C preprocessor
     *
     * save the whole kernel in variables so we don't
     * do any memory access semantics for it
     *
     *
     * this is kinda loop unrolling, and definitly register blocking
     *
     *      0  1  2
     *    ----------
     * a  |__|__|__|
     * b  |__|__|__|
     * c  |  |  |  |
     *    ----------
     */
    
    // hardcoded unflipped kernel array, fix later
    float kernel_unflipped[9] = {kernel[8], kernel[7], kernel[6], kernel[5], kernel[4], kernel[3], kernel[2], kernel[1], kernel[0]};

    __m128  kv_0, kv_1, kv_2;
            // inv_0, inv_1, inv_2, inv_4, inv_5, inv_6,
            // outv_0, outv_4;

    int x_max_stride = FLOOR_MULTIPLE(data_size_X, STRIDE);
    char needs_help  = (FLOOR_MULTIPLE(data_size_X, STRIDE) != data_size_X);


    // main convolution loop
    // avg 14 Gflops with STRIDE 8
// #pragma omp parallel for
    for (int j = 0; j < KERNY; j++){ // deal with one row of kernel at a time
        
        // load 4 copies of each column value in current kernel row into vectors
        int k0 = *(kernel_unflipped + j*KERNX + 0); // for tail
        int k1 = *(kernel_unflipped + j*KERNX + 1);
        int k2 = *(kernel_unflipped + j*KERNX + 2);
        kv_0 = _mm_load1_ps(kernel_unflipped + j*KERNX + 0); // [j0, j0, j0, j0]
        kv_1 = _mm_load1_ps(kernel_unflipped + j*KERNX + 1); // [j1, j1, j1, j1]
        kv_2 = _mm_load1_ps(kernel_unflipped + j*KERNX + 2); // [j2, j2, j2, j2]


        // avg 25-27 gflops with STRIDE 8 and no thread limit
        #pragma omp parallel for
        /*default(none) shared( j, kv_0, kv_1, kv_2, k0, k1, k2, \*/
                /*x_max_stride, data_size_X, data_size_Y, \*/
                /*out, padded, padded_width, needs_help)*/
        for(int y = j; y < data_size_Y+j; y++){ // the row of padded input
            // start y = kernel row, so 2nd kernel row isn't multiplied by 1st img row, and 3rd kernel row isn't multiplied by 1st or 2nd img row
//          #pragma omp parallel for
            for(int x = 0; x < x_max_stride; x+=STRIDE){ // x coordinate of padded input

                // the three steps of our algorithm, as macros for easily varying the step ammount
                #define LOAD( OFFSET ) \
                    __m128 outv_ ## OFFSET = _mm_loadu_ps(out + (y-j)*data_size_X + x+ (OFFSET))

                #define KERNEL_ROW( OFFSET, STORE_INTO )\
                    (STORE_INTO) = _mm_add_ps(_mm_mul_ps(kv_0, _mm_loadu_ps(padded + y*padded_width + x+0 + (OFFSET))), (STORE_INTO));\
                    (STORE_INTO) = _mm_add_ps(_mm_mul_ps(kv_1, _mm_loadu_ps(padded + y*padded_width + x+1 + (OFFSET))), (STORE_INTO));\
                    (STORE_INTO) = _mm_add_ps(_mm_mul_ps(kv_2, _mm_loadu_ps(padded + y*padded_width + x+2 + (OFFSET))), (STORE_INTO))

                #define DO( OFFSET ) KERNEL_ROW( OFFSET, outv_ ## OFFSET )

                #define STORE( OFFSET ) \
                    _mm_storeu_ps(out + (y-j)*data_size_X + x + (OFFSET), outv_ ## OFFSET)
	

                // load corresponding output block we'll sum with; all 3 input blocks sum to same output block
				LOAD(0);
                LOAD(4);

                //  multiply and sum
				DO(0);
                DO(4);

                // store into output image array
				STORE(0);
                STORE(4);
            }

            // handle tail when (data_size_X % STRIDE) != 0
            if (needs_help) {
                for(int x = x_max_stride ; x < data_size_X; x++) { 
                    float *out_index = out + (y-j)*data_size_X + x;
                    *out_index += k0 * padded[y*padded_width + x+0];
                    *out_index += k1 * padded[y*padded_width + x+1];
                    *out_index += k2 * padded[y*padded_width + x+2];

                }
            }
		}
	}

    #if USE_HEAP
    free(padded);
    #endif
	return 1;
}
