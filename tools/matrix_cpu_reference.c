// Validation-only Apple CPU reference. The inputs are already packed exactly
// in binary32; Accelerate performs each digit GEMM. Reconstruction uses exact
// dyadic arithmetic, parallelized on the CPU and storing a binary64 output.
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <stddef.h>

struct Merge {
    double *out;
    const float *low, *high, *mixed;
    size_t count;
    double radix, scale;
    int first;
};

static void merge_chunk(void *arg, size_t chunk) {
    struct Merge *ctx = (struct Merge *)arg;
    size_t begin = chunk * 16384;
    size_t end = begin + 16384 < ctx->count ? begin + 16384 : ctx->count;
    double *restrict out = ctx->out;
    const float *restrict low = ctx->low;
    const float *restrict high = ctx->high;
    const float *restrict mixed = ctx->mixed;
    const double radix = ctx->radix;
    if (!high) {
        for (size_t i = begin; i < end; ++i)
            out[i] = (double)low[i] * ctx->scale;
    } else {
        for (size_t i = begin; i < end; ++i) {
            double l = low[i], h = high[i];
            double x = (l + radix * ((double)mixed[i] - l - h) + (radix * radix) * h) * ctx->scale;
            out[i] = ctx->first ? x : out[i] + x;
        }
    }
}

void packed_cpu_gemm(const float *a, const float *b, double *out, float *temporary,
                     int m, int k, int n, int panel, int radix, int planes, double scale) {
    size_t a_plane = (size_t)m*k, b_plane = (size_t)k*n, c_plane = (size_t)m*n;
    for (int start = 0; start < k; start += panel) {
        int width = k-start < panel ? k-start : panel;
        for (int d=0; d<planes; ++d) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, width, 1.f,
                        a + d*a_plane + start, k, b + d*b_plane + (size_t)start*n,
                        n, 0.f, temporary + d*c_plane, n);
        }
        struct Merge args = {out, temporary, planes==3 ? temporary+c_plane : NULL,
                             planes==3 ? temporary+2*c_plane : NULL,
                             c_plane, (double)radix, scale, start==0};
        dispatch_apply_f((c_plane+16383)/16384, DISPATCH_APPLY_AUTO, &args, merge_chunk);
    }
}
