//#include "stdio.h"
//#include <hls_math.h>

#include "iostream"
#include "ap_int.h"
#include "ap_fixed.h"

// define the weight and feature map size
#define FM_SIZE 8
#define BN_WEIGHT_SIZE 16
#define ADD_SIZE 8
#define BN_BIAS_SIZE 32
#define ACC_SIZE 16


typedef ap_fixed<32, 16, AP_RND, AP_SAT> FIX_MAX;
// define data type
// define fm type
typedef ap_fixed<FM_SIZE, 1, AP_RND, AP_SAT> FIX_FM;

// define weight type
typedef ap_int<4> FIX_WT_4;
typedef ap_fixed<BN_WEIGHT_SIZE,3, AP_RND, AP_SAT> FIX_BN_WT;

//define add type
typedef ap_int<16> FIX_ADD_10;

//define bais type
typedef ap_fixed<BN_BIAS_SIZE, 6, AP_RND, AP_SAT> FIX_BN_BIAS;

//define result type
typedef ap_fixed<ACC_SIZE, 5, AP_RND, AP_SAT> FIX_ACC;


//define compute part
template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT, typename FIX_BIAS>
void batchnorm(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_BIAS *bias,
               int CHANNEL_IN, int HEIGHT, int WIDTH);
template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT, typename FIX_ADD>
void downsample(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output,
                FIX_ADD *add, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH);
template <typename FIX_INPUT, typename FIX_OUTPUT>
void relu(FIX_INPUT *input, FIX_OUTPUT *output, int size);

//define con3x3*stride2
template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_ADD, typename FIX_OUTPUT>
void conv3x3_stride2(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_ADD *add, FIX_OUTPUT *output, FIX_OUTPUT *result,
                     FIX_INPUT *input_padding, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH);

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
void shift_stride2(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_INPUT *input_padding,
                   int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH);

//define con3x3*stride1
template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_ADD, typename FIX_OUTPUT>
void conv3x3_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_ADD *add, FIX_OUTPUT *output, FIX_OUTPUT *result,
                     FIX_INPUT *input_padding, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH);

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
void shift_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_INPUT *input_padding,
                   int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH);

template <typename FIX_INPUT, typename FIX_ADD, typename FIX_OUTPUT>
void adder(FIX_INPUT *input, FIX_INPUT *input_padding, FIX_ADD *add, FIX_OUTPUT *output, int CHANNEL_OUT, int HEIGHT, int WIDTH);

void shiftadder(FIX_FM input[3 * 32 * 32], FIX_WT_4 weight[74160], FIX_ADD_10 add[96768], FIX_BN_WT bn_wt[336],
                FIX_BN_BIAS bn_bias[336], int r[1]);
