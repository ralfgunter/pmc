// TODO: should I worry about endianess?
#include "bitset2d.h"
#include <stdio.h>

// Host functions
Bitset2D bitset_new(uint32_t dim_x, uint32_t dim_y)
{
    Bitset2D b;
    uint32_t num_y = (dim_y + BITS_PER_CHAR - 1) / BITS_PER_CHAR;
    uint32_t num_elements = dim_x * num_y;
    uint32_t *matrix = (uint32_t *) calloc(num_elements, sizeof(uint32_t));

    b.matrix = matrix;
    b.num_x = dim_x;
    b.num_y = num_y;

    return b;
}

void bitset_free(Bitset2D b)
{
    free(b.matrix);
}

uint32_t bitset_size(Bitset2D b)
{
    return b.num_x * b.num_y;
}

uint32_t bitset_get(Bitset2D b, uint32_t x, uint32_t y)
{
    return (b.matrix[MATRIX_IDX(x,y,b.num_y)] >> UINT_IDX(y)) & 0x01;
}
