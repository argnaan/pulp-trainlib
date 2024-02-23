/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pulp_train.h"

#include "init-defines.h"
#include "input-sequence.h"
#include "skip-output.h"
#include "stats.h"

#include "step-check.h"
#include "stats.h"

#include "net.h"



// DATA DEFINITION


// GELU
PI_L1 fp16 zero_init = 0.0f;
PI_L1 struct SkipConn_args_fp16 skip_args;
PI_L1 struct blob_fp16 layer0_in_1, layer0_in_2, layer0_out;

// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

#ifdef FORWARD
PI_L1 fp16 l0_in_1[INPUT_SIZE];
PI_L1 fp16 l0_in_2[INPUT_SIZE];
PI_L1 fp16 l0_out[INPUT_SIZE];
#endif

#ifdef BACKWARD
#endif



#ifdef FORWARD
static inline void tensor_init() 
{
  printf("Initializing the things\n");
  for (int i=0; i<INPUT_SIZE; i++)               l0_in_1[i] = INPUT_1[i];
  for (int i=0; i<INPUT_SIZE; i++)               l0_in_2[i] = INPUT_2[i];
  for (int i=0; i<INPUT_SIZE; i++)               l0_out[i] = zero_init; 
  printf("Finished initializing the things\n");
}

static inline void connect_blobs() 
{
  layer0_in_1.data = l0_in_1;
  layer0_in_1.dim = INPUT_SIZE;
  layer0_in_1.W = Tin_W_l1;
  layer0_in_1.H = Tin_H_l1;
  layer0_in_1.C = Tin_C_l1;

  layer0_in_2.data = l0_in_2;
  layer0_in_2.dim = INPUT_SIZE;
  layer0_in_2.W = Tin_W_l1;
  layer0_in_2.H = Tin_H_l1;
  layer0_in_2.C = Tin_C_l1;

  layer0_out.data = l0_out;
  layer0_out.dim = INPUT_SIZE;
  layer0_out.H = Tin_H_l1;
  layer0_out.W = Tin_W_l1;
  layer0_out.C = Tin_C_l1;

  skip_args.skip = &layer0_in_1;
  skip_args.lout = &layer0_in_2;
  skip_args.output = &layer0_out;
}

static inline void compute_memory_occupation(){
  // Input 1
  L1_memocc_bytes += INPUT_SIZE *sizeof(fp16);
  // Input 2
  L1_memocc_bytes += INPUT_SIZE *sizeof(fp16);
  // Output
  L1_memocc_bytes += INPUT_SIZE*sizeof(fp16);

  // Input 1
  L2_memocc_bytes += INPUT_SIZE *sizeof(fp16);
  // Input 2
  L2_memocc_bytes += INPUT_SIZE *sizeof(fp16);
  // Output
  L2_memocc_bytes += INPUT_SIZE*sizeof(fp16);
}
#endif


#ifdef BACKWARD
#endif



static inline void compare_tensors(fp16 *A, fp16 *B, int length){
  fp16 mean_err_rel = zero_init;
  fp16 diff = zero_init;

  for(int i=0; i<length; i++){
    diff = A[i]-B[i];
    if (diff>0) diff = diff;
    else diff=-diff;
    mean_err_rel = mean_err_rel + diff/length;
  }
  if (mean_err_rel<ERROR_TOLERANCE) printf("\n>>>TENSOR MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
  else printf("\n>>>TENSOR NOT MATCHING!\nMEAN ERROR:%f\n", mean_err_rel);
}

// Elementwise checker
int check_tensor(fp16 * tensor_out, fp16 * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



static inline void train(){
  pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR)  | (1<<PI_PERF_LD)  | (1<<PI_PERF_ACTIVE_CYCLES) );
  pi_perf_stop();
  pi_perf_reset();
  pi_perf_start();
 
  #ifdef PROF_FWD
  printf("\nForward stats\n");
  START_STATS();
  #endif

  #ifdef FORWARD
  pulp_residualconn_fp16_fw(&skip_args);
  #endif

  #ifdef PROF_FWD
  STOP_STATS();
  #endif

  #ifdef BACKWARD
  #endif

  pi_perf_stop();

  int instr_count=pi_perf_read (PI_PERF_INSTR);
  int cycles_count=pi_perf_read (PI_PERF_CYCLES);
  int load_count=pi_perf_read (PI_PERF_LD);
  int active_cycles_count=pi_perf_read (PI_PERF_ACTIVE_CYCLES);

  printf("\nperformance\n");
  printf("\nCycles count %d \n", cycles_count);
  printf("Instruction Count %d\n", instr_count);
  printf("Active Cycles Count %d\n", active_cycles_count);
  printf("Load Count %d\n", load_count);
  printf("Cycles/Instruction %f\n", (fp16)cycles_count/instr_count);
  


  #ifdef FORWARD
  printf("\nFORWARD CHECK: \n");
  compare_tensors(l0_out, OUTPUT, OUTPUT_SIZE);
  check_tensor(l0_out, OUTPUT, OUTPUT_SIZE);
  #endif


  #ifdef BACKWARD       
  #endif 
}


// Most important function: it connects each passage to step the net and perform training
void net_step()
{
  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  #endif

  #ifdef MEMOCC_COMP
  compute_memory_occupation();
  printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
  printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
  #endif

  tensor_init();

  connect_blobs();

  train();

  return;
}
