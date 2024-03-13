#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_rmsnorm_fp32.h"
#include "math.h"

void rmsnorm_parallelized(float* o, float* x, float* weight, int size){
    struct sum_of_squares_args ss_args;
    ss_args.in = x;
    ss_args.out = o; // si sfrutta come buffer il buffer di uscita o. Funziona solo se size >= NUM_CORES e se non è in-place
    ss_args.size = size;
    
    pi_cl_team_fork(NUM_CORES, sum_of_squares, &ss_args);

    float ss = 0;
    for(int i=0; i<NUM_CORES; i++)
        ss += o[i];

    ss /= size;
    ss += 1e-5f;

    #ifdef Q_RSQRT
    ss = q_rsqrt(ss);
    #else
    ss = 1.0f / sqrtf(ss);
    #endif

    struct weighted_scaling_args ws_args;
    ws_args.in = x;
    ws_args.out = o;
    ws_args.w = weight;
    ws_args.size = size;
    ws_args.scaling_factor = ss;

    pi_cl_team_fork(NUM_CORES, weighted_scaling, &ws_args);
}

void weighted_scaling(void* weighted_scaling_args) {
    struct weighted_scaling_args* args = (struct weighted_scaling_args* )weighted_scaling_args;
    float* out = args->out;
    float* in = args->in;
    float* w = args->w;
    float sf = args->scaling_factor;
    unsigned int size = args->size;

    const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > size ? size : start+blockSize;

    for (uint32_t i = start; i < stop; i++) {
        out[i] = w[i] * (sf * in[i]);
    }
}

void sum_of_squares(void* ss_args){
    struct sum_of_squares_args* args = (struct sum_of_squares_args*)ss_args;
    float* out = args->out;
    float* in = args->in;
    unsigned int size = args->size;

    int id = pi_core_id();

    const uint32_t blockSize = (size+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = id*blockSize;
    const uint32_t stop = start+blockSize > size ? size : start+blockSize;

    out[id] = 0;
    for (uint32_t i = start; i < stop; i++) {
        out[id] += in[i] * in[i];
    }
}