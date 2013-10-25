#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
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
            if (y-1 > -1) {
                if (x-1 > -1) cur_sum += in[x-1 + (y-1) * data_size_X] * k_a0;
                              cur_sum += in[x   + (y-1) * data_size_X] * k_a1;
                if (x+1 < dX) cur_sum += in[x+1 + (y-1) * data_size_X] * k_a2;
            }

            // center row
            if (x-1 > -1) cur_sum += in[x-1 + (y  ) * data_size_X] * k_b0;
                          cur_sum += in[x   + (y  ) * data_size_X] * k_b1;
            if (x+1 < dX) cur_sum += in[x+1 + (y  ) * data_size_X] * k_b2;

            // bottom row
            if (y+1 < dY) {
                if (x-1 > -1) cur_sum += in[x-1 + (y+1) * data_size_X] * k_c0;
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
