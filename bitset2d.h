#ifndef _BITSET2D_H_
#define _BITSET2D_H_

#include <stdlib.h>
#include <stdint.h>

#define BITS_PER_UINT (sizeof(uint32_t) * 8)

#define MATRIX_IDX(x,y,num_y) ((y / BITS_PER_UINT) + (x * num_y))
#define UINT_IDX(y)            (y % BITS_PER_UINT)

typedef struct {
    uint32_t *matrix;
    uint32_t num_x, num_y;
} Bitset2D;

extern Bitset2D bitset_new(uint32_t dim_x, uint32_t dim_y);
extern uint32_t bitset_get(Bitset2D b, uint32_t x, uint32_t y);
extern uint32_t bitset_size(Bitset2D b);
extern void bitset_free(Bitset2D b);

#endif // _BITSET2D_H_
