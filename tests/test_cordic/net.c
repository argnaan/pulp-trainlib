#include "pmsis.h"
#include "pulp_train.h"

#include "net.h"
#include "stats.h"
#include "cordic_data.h"
#include "math.h"

PI_L1 float cosines[N_TEST];
PI_L1 float sines[N_TEST];

void net_step () {
    printf("Using standard math.h cosf and sinf:\n");
    
    INIT_STATS();
    PRE_START_STATS();
    START_STATS();

    for(int i=0;i<N_TEST;i++){
        cosines[i] = cosf(gm_angles[i]);
        sines[i] = sinf(gm_angles[i]);
    }

    STOP_STATS();

    printf("\n%10s %10s %10s %10s %10s\n", "angle", "gm_cos", "cosf", "gm_sin", "sinf");
    for(int i=0;i<N_TEST;i++)
        printf("%10f %10f %10f %10f %10f\n", gm_angles[i], gm_cos[i], cosines[i], gm_sin[i], sines[i]);

    printf("\n\nUsing cordic cos and sin: \n");

    START_STATS();

    for(int i=0;i<N_TEST;i++)
        cordic_cos_sin_fp32(gm_angles[i], &cosines[i], &sines[i]);

    STOP_STATS();

    printf("\n%10s %10s %10s %10s %10s\n", "angle", "gm_cos", "cordic cos", "gm_sin", "cordic sin");
    for(int i=0;i<N_TEST;i++)
        printf("%10f %10f %10f %10f %10f\n", gm_angles[i], gm_cos[i], cosines[i], gm_sin[i], sines[i]);

    printf("\n\nTest finished\n\n");
    return;
}