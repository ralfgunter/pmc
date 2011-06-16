// TODO: should I worry about endianess?
#include "bitset2d.h"
#include <stdio.h>

// Host functions
Bitset2D bitset_new(uint dim_x, uint dim_y)
{
    Bitset2D b;
    uint num_y = (dim_y + BITS_PER_UINT - 1) / BITS_PER_UINT;
    uint num_elements = dim_x * num_y;
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

uint bitset_size(Bitset2D b)
{
    return b.num_x * b.num_y;
}

uint bitset_get(Bitset2D b, uint x, uint y)
{
    return (b.matrix[MATRIX_IDX(x,y,b.num_y)] >> UINT_IDX(y)) & 0x01;
}
