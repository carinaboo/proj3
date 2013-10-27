#include <emmintrin.h>
#include <string.h>
#include <stdio.h>
// string.h for memset
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

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

    float *padded = malloc(p_arr_size);

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
    for (int i = 0; i < out.height; i++) {
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

/* new pseudocode:
(obv don't do as function calls, its just to show you the idea)
this is to do the  one-kernel-cell at a time algo with 9 hardcoded
kernel vectors and no padding at all!!!!
def do_row(kernel_row, image_y) {
	x = 0
	do_column(0, x)
	do_column(1, x)
	// don't do 3rd col because it would write out of bounds
	for (x = 1; x < image_data_X-1; x++) {
		do_column(0, x)
		do_column(1, x)
		do_column(2, x)
	}
	x++
	do_column(1, x)
	do_column(2, x)
}

// here's the main body
y = 0
do_row(a, y)
do_row(b, y)
// don't do c, y because it would write out of bounds
for (y = 1; y < image_data_Y-1; y++) {
	do_row(a,b,c, 0)
}
y++
do_row(b, y)
do_row(c, y)
}


: D
*/

// assumes float *in, float *out, data_size_X, data_size_Y
// kernel vecs kv_[a-z][0-9]*
// __mm128 inv input-array vector containing X+0 thru X+3 from this row
// TODO write this after i figure out my prelude
// #define DO_COLUMN( ROW, COLUMN, X) return X

#define STRIDE 4
/*// a0. x+1, y+1
_mm_store_ps(in_origin + 1 + data_size_Y,
        _mm_add_ps(
            _mm_mul_ps(kv_a0, in_v),
            _mm_load_ps(in_origin + 1 + data_size_Y)));
            */
// assumes data_size_Y
// KERN_ROW is a,b,c
// KERN_COL is 0,1,2
// assumes *float in
// assumes *float out
#define ROW_OFFSET_a data_size_X
#define ROW_OFFSET_b 0
#define ROW_OFFSET_c (-data_size_X)
#define VECT_CONV( KERN_ROW, KERN_COL, IN_VEC, OFFSET ) \
    _mm_storeu_ps(out + (OFFSET) + (1-(KERN_COL)) +  ROW_OFFSET_##KERN_ROW, \
        _mm_add_ps( \
                _mm_mul_ps( kv_##KERN_ROW##KERN_COL , (IN_VEC)), \
                _mm_loadu_ps(out + (OFFSET) + (1-(KERN_COL)) + ROW_OFFSET_##KERN_ROW)))

// assumes *float out
#define ROW_OFFSET_PADDED_a (pad_w)
#define ROW_OFFSET_PADDED_b 0
#define ROW_OFFSET_PADDED_c (-pad_w)
#define VECT_CONV_PADDED( KERN_ROW, KERN_COL, IN_VEC, OFFSET ) \
    _mm_storeu_ps(padded + (OFFSET) + (1-(KERN_COL)) +  ROW_OFFSET_PADDED_##KERN_ROW, \
        _mm_add_ps( \
                _mm_mul_ps( kv_##KERN_ROW##KERN_COL , (IN_VEC)), \
                _mm_loadu_ps(padded + (OFFSET) + (1-(KERN_COL)) + ROW_OFFSET_PADDED_##KERN_ROW)))

#define RUN_COLUMN( KERN_ROW, KERN_COL) \
        for(x = 0; x < x_stride_max; x+=STRIDE) { \
            offset = x + start_of_row; \
            p_offset = x + p_start_of_row; \
            in_v = _mm_loadu_ps(in+offset); \
            VECT_CONV_PADDED(KERN_ROW, KERN_COL, in_v, p_offset); \
        } \
        for ( ; x < data_size_X; x++) { \
            offset = x + start_of_row; \
            p_offset = x + p_start_of_row; \
            out[p_offset + (1-(KERN_COL)) + ROW_OFFSET_##KERN_ROW] += in[offset] * k_##KERN_ROW##0; \
        }\

#define RUN_KERNEL( KERN_ROW ) \
    RUN_COLUMN( KERN_ROW, 0 ); \
    RUN_COLUMN( KERN_ROW, 1 ); \
    RUN_COLUMN( KERN_ROW, 2 ); \

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

    // float kernel - for when we're outside of our stride
    k_a0 = *(kernel + 2 + 2*KERNX);
    k_a1 = *(kernel + 1 + 2*KERNX);
    k_a2 = *(kernel + 0 + 2*KERNX);
    k_b0 = *(kernel + 2 + 1*KERNX);
    k_b1 = *(kernel + 1 + 1*KERNX);
    k_b2 = *(kernel + 0 + 1*KERNX);
    k_c0 = *(kernel + 2 + 0*KERNX);
    k_c1 = *(kernel + 1 + 0*KERNX);
    k_c2 = *(kernel + 0 + 0*KERNX);

    // vector kernel - gotta go fast
    __m128 kv_a0 = _mm_load1_ps(&k_a0);
    __m128 kv_a1 = _mm_load1_ps(&k_a1);
    __m128 kv_a2 = _mm_load1_ps(&k_a2);
    __m128 kv_b0 = _mm_load1_ps(&k_b0);
    __m128 kv_b1 = _mm_load1_ps(&k_b1);
    __m128 kv_b2 = _mm_load1_ps(&k_b2);
    __m128 kv_c0 = _mm_load1_ps(&k_c0);
    __m128 kv_c1 = _mm_load1_ps(&k_c1);
    __m128 kv_c2 = _mm_load1_ps(&k_c2);

    array2d pad;
    array2d out2d;
    out2d.width = data_size_X;
    pad.width = data_size_X+2;
    out2d.height = data_size_Y;
    pad.height = data_size_Y+2;

    float padded[ (data_size_X+2) * (data_size_Y+2) ];
    memset(padded, 0, (data_size_X+2) * (data_size_Y+2) * sizeof(float));
    pad.array = padded;

    out2d.array = out;


    int pad_w = pad.width;
    int x_stride_max = ((data_size_X)/STRIDE)*STRIDE;
    int offset;
    int p_offset;
    int start_of_row;
    int p_start_of_row;

    int x;

    // current in-array 4-tuple
    __m128 in_v;

    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on

        start_of_row = y * data_size_X;
        p_start_of_row = (y+1) * pad_w + 1;

        RUN_KERNEL( a );
        RUN_KERNEL( b );
        RUN_KERNEL( c );
	}

    unPad(pad, out2d, 1);

    // free the padded matrix
	return 1;
}
