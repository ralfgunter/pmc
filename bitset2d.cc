// TODO: should I worry about endianess?
#include "bitset2d.h"
#include <stdio.h>

// Host functions
Bitset2D bitset_new(int dim_x, int dim_y)
{
    Bitset2D b;
    int num_y = (dim_y + BITS_PER_UINT - 1) / BITS_PER_UINT;
    int num_elements = dim_x * num_y;
    uint *matrix = (uint *) calloc(num_elements, sizeof(uint));

    b.matrix = matrix;
    b.num_x = dim_x;
    b.num_y = num_y;

    return b;
}

void bitset_free(Bitset2D b)
{
    free(b.matrix);
}

int bitset_size(Bitset2D b)
{
    return b.num_x * b.num_y;
}

int bitset_get(Bitset2D b, int x, int y)
{
    return (b.matrix[MATRIX_IDX(x,y,b.num_y)] >> UINT_IDX(y)) & 0x01;
}
