#include "genCases.h"
#include <stdlib.h>

uint64_t begin_cases(int op, int level) {
    srand(1);
    genCases_setLevel(level);
    softfloat_roundingMode=softfloat_round_near_even;
    if(op==4) genCases_f64_a_init();
    else if(op==5) genCases_f64_abc_init();
    else genCases_f64_ab_init();
    return genCases_total;
}

void make_cases(int op, uint32_t count,uint64_t *a,uint64_t *b,uint64_t *c,uint64_t *out) {
    for(uint32_t i=0;i<count;i++) {
        if(op==4) genCases_f64_a_next();
        else if(op==5) genCases_f64_abc_next();
        else genCases_f64_ab_next();
        float64_t x=genCases_f64_a,y=genCases_f64_b,z=genCases_f64_c,r;
        a[i]=x.v;b[i]=y.v;c[i]=z.v;
        switch(op) {
        case 0:r=f64_add(x,y);break;
        case 1:r=f64_sub(x,y);break;
        case 2:r=f64_mul(x,y);break;
        case 3:r=f64_div(x,y);break;
        case 4:r=f64_sqrt(x);break;
        default:r=f64_mulAdd(x,y,z);break;
        }
        out[i]=r.v;
    }
}
