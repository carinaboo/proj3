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

    // pad the array with a ring of zeroes so we don't have to stress about dis shiz
    // using padding instead of ifs all over the place got us like ~1.5gflops
    array2d in_2d;
    in_2d.array = in;
    in_2d.width = data_size_X;
    in_2d.height = data_size_Y;
    array2d pad_2d = zeroPad(in_2d, 1);
    float* padded = pad_2d.array;
    int pad_width = pad_2d.width;


    // accumulator so we don't access deep array memory every multiply.
    float cur_sum = 0;

    int ya, yb, yc; // y-1*width, y*width, y+1*width
    
    // main convolution loop
    // reording y before x got us 4gflops
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
        for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
            // re-initialize sum
            cur_sum = 0;

            // maybe register blocking these will speed things up?
            // it didn't
            ya = (y) * pad_width;
            yb = (y+1) * pad_width;
            yc = (y+2) * pad_width;

            // for now i'm not handling top/bottom/left/right errors
            // because it's a lot of if statements
            // also note that the kernel is NOT flipped -- woo doing intuitive things
            cur_sum += padded[x   + ya] * k_a0;
            cur_sum += padded[x+1 + ya] * k_a1;
            cur_sum += padded[x+2 + ya] * k_a2;

            cur_sum += padded[x   + yb] * k_b0;
            cur_sum += padded[x+1 + yb] * k_b1;
            cur_sum += padded[x+2 + yb] * k_b2;

            cur_sum += padded[x   + yc] * k_c0;
            cur_sum += padded[x+1 + yc] * k_c1;
            cur_sum += padded[x+2 + yc] * k_c2;


            // store into out matrix
            out[x+y*data_size_X] = cur_sum;
            /*
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}*/
		}
	}

    // free the padded matrix
    free(padded);
	return 1;
}
