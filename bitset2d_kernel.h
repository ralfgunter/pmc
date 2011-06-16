#include "bitset2d.h"

__device__ void gpu_set(Bitset2D b, uint x, uint y)
{
    b.matrix[MATRIX_IDX(x,y,b.num_y)] |= (1 << UINT_IDX(y));
}
