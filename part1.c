#include <emmintrin.h>
#include <string.h>
#include <stdio.h>
// string.h for memset
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define STRIDE 4 // gotta go fast

typedef struct {
    int width;
    int height;
    float *array;
} array2d;

// add some number of zeros as padding around a given strided-array
// for a 3x3 kernel, the pad_size should be one
// in general pad_size = max((KERNX - 1)/2, (KERNY - 1)/2);
// note that we'll need to free whatever memory is in *padded elsewhere!
array2d zeroPad(array2d in, int pad_size) {
    array2d retval;
    int pad_x = in.width + pad_size*2;
    int pad_y = in.height + pad_size*2;

    size_t p_arr_size = pad_x*pad_y*sizeof(float);
    size_t line_size = in.width*sizeof(float);

    float *padded = (float*) malloc(p_arr_size);

    // zero out the whole dingus
    memset(padded, 0, p_arr_size);

    // copy the original data into the zero-padded array
    for (int i = in.height-1; i != -1; i--) {
        //     0,0      y                   x         
        memcpy(padded + pad_x*(i+pad_size) + pad_size,
                // 0,y
                in.array + i*in.width, 
                line_size);
    }

    retval.array = padded;
    retval.width = pad_x;
    retval.height = pad_y;

    return retval;
}

// unpads padded.array by pad_size from all size
// writes result into out.array (which should be *out from 
// conv2d :)
void unPad(array2d padded, array2d out, int pad_size) {
    // copy data out of the zero-padded array
    float *src = padded.array;
    float *dst = out.array;
    size_t line_size = out.width*sizeof(float);
    for (int i = out.height-1; i != -1; i--) {
        memcpy(dst + i*out.width,
               src + padded.width*(i+pad_size) + pad_size,
               line_size);
    }
}


// tool for debugging zeroPad
void printArray(array2d array) {
    for (int y = 0; y < array.height; y++) {
        for (int x = 0; x < array.width; x++) {
            printf("%f, ", array.array[x + y*array.width]);
        }
        printf("\n");
    }
}

// test the above functions on arrays
void test_array2d(int pad_size) {
    float data[9] = {1,2,3,4,5,6,7,8,9};
    float res[9];
    array2d padded;
    array2d out;
    array2d test;

    test.array = data;
    test.width = 3;
    test.height = 3;

    out.array = res;
    out.width = 3;
    out.height = 3;


    printf("Original data\n");
    printArray(test);

    printf("Padded with %i 0s on each side\n", pad_size);
    padded = zeroPad(test, pad_size);
    printArray(padded);

    printf("Padding removed!\n");
    unPad(padded, out, pad_size);
    printArray(out);
}

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    size_t float_size = sizeof(float);
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    /* assuming KERNX = 3 and KERNY = 3,
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

    // register-blocking the kernel got us like ~2.5gflops
    // along with regoster-blocking the current sum
    float k_a0, k_a1, k_a2, 
          k_b0, k_b1, k_b2,
          k_c0, k_c1, k_c2;

    // kernel un-flipped because why would you do that to me
    k_a0 = *(kernel + 2 + 2*KERNX);
    k_a1 = *(kernel + 1 + 2*KERNX);
    k_a2 = *(kernel + 0 + 2*KERNX);
    k_b0 = *(kernel + 2 + 1*KERNX);
    k_b1 = *(kernel + 1 + 1*KERNX);
    k_b2 = *(kernel + 0 + 1*KERNX);
    k_c0 = *(kernel + 2 + 0*KERNX);
    k_c1 = *(kernel + 1 + 0*KERNX);
    k_c2 = *(kernel + 0 + 0*KERNX);

    // kernel vectorized
    // later, we should make any size kernel work
    float k_a[4] = {k_a0,k_a1,k_a2,0};
    float k_b[4] = {k_b0,k_b1,k_b2,0};
    float k_c[4] = {k_c0,k_c1,k_c2,0};
    __m128 kv_a = _mm_loadu_ps(k_a);
    __m128 kv_b = _mm_loadu_ps(k_b);
    __m128 kv_c = _mm_loadu_ps(k_c);


    // pad the array with a ring of zeroes so we don't have to stress about dis shiz
    // using padding instead of ifs all over the place got us like ~1.5gflops
    array2d in_2d;
    in_2d.array = in;
    in_2d.width = data_size_X;
    in_2d.height = data_size_Y;
    array2d pad_2d = zeroPad(in_2d, 1);
    float* padded = pad_2d.array;
    int pad_width = pad_2d.width;


    // accumulators so we don't access deep array memory every multiply.
    float cur_sum0 = 0;
    float cur_sum1 = 0;
    float cur_sum2 = 0;
    float cur_sum3 = 0;
    float sum_arr[16];

    // vector registers that we will load in data
    __m128 sum_v0, sum_v1, sum_v2, sum_v3, 
           inv_a0, inv_a1, inv_a2, inv_a3, 
           inv_b0, inv_b1, inv_b2, inv_b3, 
           inv_c0, inv_c1, inv_c2, inv_c3;

    // int ya, yb, yc; // y-1*width, y*width, y+1*width
    
    // main convolution loop
    // reording y before x got us 4gflops
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
        for(int x = 0; x < data_size_X/STRIDE*STRIDE; x+=STRIDE){ // the x coordinate of the output location we're focusing on
            // STRIDING
            cur_sum0 = 0;
            cur_sum1 = 0;
            cur_sum2 = 0;
            cur_sum3 = 0;
            sum_v0 = _mm_setzero_ps();
            sum_v1 = _mm_setzero_ps();
            sum_v2 = _mm_setzero_ps();
            sum_v3 = _mm_setzero_ps();

            // current input block vectorized; last column of will be 0s when multiplied by kernel
            inv_a0 = _mm_loadu_ps(padded + x+0 + y*pad_width);
            inv_b0 = _mm_loadu_ps(padded + x+0 + (y+1)*pad_width);
            inv_c0 = _mm_loadu_ps(padded + x+0 + (y+2)*pad_width);
            inv_a1 = _mm_loadu_ps(padded + x+1 + y*pad_width);
            inv_b1 = _mm_loadu_ps(padded + x+1 + (y+1)*pad_width);
            inv_c1 = _mm_loadu_ps(padded + x+1 + (y+2)*pad_width);
            inv_a2 = _mm_loadu_ps(padded + x+2 + y*pad_width);
            inv_b2 = _mm_loadu_ps(padded + x+2 + (y+1)*pad_width);
            inv_c2 = _mm_loadu_ps(padded + x+2 + (y+2)*pad_width);
            inv_a3 = _mm_loadu_ps(padded + x+3 + y*pad_width);
            inv_b3 = _mm_loadu_ps(padded + x+3 + (y+1)*pad_width);
            inv_c3 = _mm_loadu_ps(padded + x+3 + (y+2)*pad_width);

            // multiply kernel with current input block
            inv_a0 = _mm_mul_ps(kv_a, inv_a0);
            inv_b0 = _mm_mul_ps(kv_b, inv_b0);
            inv_c0 = _mm_mul_ps(kv_c, inv_c0);
            inv_a1 = _mm_mul_ps(kv_a, inv_a1);
            inv_b1 = _mm_mul_ps(kv_b, inv_b1);
            inv_c1 = _mm_mul_ps(kv_c, inv_c1);
            inv_a2 = _mm_mul_ps(kv_a, inv_a2);
            inv_b2 = _mm_mul_ps(kv_b, inv_b2);
            inv_c2 = _mm_mul_ps(kv_c, inv_c2);
            inv_a3 = _mm_mul_ps(kv_a, inv_a3);
            inv_b3 = _mm_mul_ps(kv_b, inv_b3);
            inv_c3 = _mm_mul_ps(kv_c, inv_c3);

            // summing
            sum_v0 = _mm_add_ps(inv_a0, sum_v0);
            sum_v0 = _mm_add_ps(inv_b0, sum_v0);
            sum_v0 = _mm_add_ps(inv_c0, sum_v0);
            sum_v1 = _mm_add_ps(inv_a1, sum_v1);
            sum_v1 = _mm_add_ps(inv_b1, sum_v1);
            sum_v1 = _mm_add_ps(inv_c1, sum_v1);
            sum_v2 = _mm_add_ps(inv_a2, sum_v2);
            sum_v2 = _mm_add_ps(inv_b2, sum_v2);
            sum_v2 = _mm_add_ps(inv_c2, sum_v2);
            sum_v3 = _mm_add_ps(inv_a3, sum_v3);
            sum_v3 = _mm_add_ps(inv_b3, sum_v3);
            sum_v3 = _mm_add_ps(inv_c3, sum_v3);

            // read out vector data and perform final addition
            _mm_storeu_ps(sum_arr+ 0, sum_v0);
            _mm_storeu_ps(sum_arr+ 4, sum_v1);
            _mm_storeu_ps(sum_arr+ 8, sum_v2);
            _mm_storeu_ps(sum_arr+12, sum_v3);

            // should we manually unroll this, or is compiler smart?
            for (int i = 0; i < 3; i++) {
                cur_sum0 += sum_arr[i];
                cur_sum1 += sum_arr[i+4];
                cur_sum2 += sum_arr[i+8];
                cur_sum3 += sum_arr[i+12];
            }

            // store into out matrix
            out[x+0+y*data_size_X] = cur_sum0;
            out[x+1+y*data_size_X] = cur_sum1;
            out[x+2+y*data_size_X] = cur_sum2;
            out[x+3+y*data_size_X] = cur_sum3;
		}
	}

    // free the padded matrix
    free(padded);
	return 1;
}
