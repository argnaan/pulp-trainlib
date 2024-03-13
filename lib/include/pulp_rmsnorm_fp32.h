

struct weighted_scaling_args {
    float* out;
    float* in;
    float* w;
    float scaling_factor;
    unsigned int size;
};

struct sum_of_squares_args {
    float* out;
    float* in;
    unsigned int size;
};

void weighted_scaling(void* weighted_scaling_args);

void sum_of_squares(void* sum_of_squares_args);

void rmsnorm_parallelized(float* o, float* x, float* weight, int size);
