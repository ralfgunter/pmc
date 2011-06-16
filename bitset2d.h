#ifndef _BITSET2D_H_
#define _BITSET2D_H_

#include <stdlib.h>

#define BITS_PER_UINT (sizeof(uint) * 8)

#define MATRIX_IDX(x,y,num_y) ((y / BITS_PER_UINT) + (x * num_y))
#define UINT_IDX(y)            (y % BITS_PER_UINT)

typedef struct {
    uint *matrix;
    uint num_x, num_y;
} Bitset2D;

extern Bitset2D bitset_new(uint dim_x, uint dim_y);
extern uint bitset_get(Bitset2D b, uint x, uint y);
extern uint bitset_size(Bitset2D b);
extern void bitset_free(Bitset2D b);

#endif // _BITSET2D_H_
