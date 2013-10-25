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
        memcpy(padded + pad_x*(i+1) + pad_size, in.array + i*in.width, line_size);
    }

    retval.array = padded;
    retval.width = pad_x;
    retval.height = pad_y;

    return retval;
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


int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{

    float test_data[9] = {1,2,3,4,5,6,7,8,9,};
    array2d test;
    test.array = test_data;
    test.width = 3;
    test.height = 3;

    // first print in2d
    printArray(test);

    // now get a copy with padding
    array2d padded = zeroPad(test, 1);

    // and print that
    printArray(padded);

    // just for testing for now !!!
    return 1;

    size_t float_size = sizeof(float);
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;

    // padded array
    int padded_count = (data_size_X+2)*(data_size_Y+2);
    float p[padded_count];
    // initialize it all to zero
    // memset(ptr, 0, number of bytes);
    memset(p, 0, padded_count*float_size);


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

    float k_a0, k_a1, k_a2, 
          k_b0, k_b1, k_b2,
          k_c0, k_c1, k_c2;

    int dX = data_size_X;
    int dY = data_size_Y;

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

    // accumulator so we don't access deep array memory every multiply.
    float cur_sum = 0;
    
    // main convolution loop
	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
            // re-initialize sum
            cur_sum = 0;

            // for now i'm not handling top/bottom/left/right errors
            // because it's a lot of if statements
            // also note that the kernel is NOT flipped -- woo doing intuitive things

            // first row, one above the center
            if (y != 0) {
                if (x != 0)   cur_sum += in[x-1 + (y-1) * data_size_X] * k_a0;
                              cur_sum += in[x   + (y-1) * data_size_X] * k_a1;
                if (x+1 < dX) cur_sum += in[x+1 + (y-1) * data_size_X] * k_a2;
            }

            // center row
            if (x != 0)   cur_sum += in[x-1 + (y  ) * data_size_X] * k_b0;
                          cur_sum += in[x   + (y  ) * data_size_X] * k_b1;
            if (x+1 < dX) cur_sum += in[x+1 + (y  ) * data_size_X] * k_b2;

            // bottom row
            if (y+1 < dY) {
                if (x != 0)   cur_sum += in[x-1 + (y+1) * data_size_X] * k_c0;
                              cur_sum += in[x   + (y+1) * data_size_X] * k_c1;
                if (x+1 < dX) cur_sum += in[x+1 + (y+1) * data_size_X] * k_c2;
            }

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
	return 1;
}
