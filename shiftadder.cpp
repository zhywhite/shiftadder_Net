#include "shiftadder.h"
#include "stdlib.h"
using namespace std;
// compute the add/weight size
/* weight1 16*3*3*3 
bn_w 16 bn_b 16

weight_layer1_1 16*16*3*3  add_layer1_1 16*16*3*3
bn_w 16 bn_b 16
weight_layer1_2 16*16*3*3  add_layer1_2 16*16*3*3
bn_w 16 bn_b 16

weight_layer2_1 32*16*3*3 add_layer2_1 32*32*3*3
bn_w 32 bn_b 32
weight_layer2_2 32*32*3*3 add_layer2_2 32*32*3*3
bn_w 32 bn_b 32
downsaple wt 32*16*3*3 add 32*32*3*3
bn_w 32 bn_b 32

weight_layer3_1 64*32*3*3 add_layer2_1 64*64*3*3
bn_w 64 bn_b 64
weight_layer2_2 64*64*3*3 add_layer2_2 64*64*3*3
bn_w 64 bn_b 64
downsaple wt 64*32*3*3 add 64*64*3*3
bn_w 64 bn_b 64

fc_weight 64*num*/

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
FIX_OUTPUT shift(FIX_INPUT input, FIX_WEIGHT weight)
{
    if (weight == 0)
    {
        return 0;
    }
    else
    {
        if (weight > 0)
        {
            return (input >> weight);
        }
        else
            return (-(input >>(-weight)));
    }
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
void shift_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_INPUT *input_padding,
                   int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    FIX_OUTPUT result = 0;
    //clear input_buf
    int i, j, k, index = 0, index_padding = 0;
    // padding
    for (k = 0; k < CHANNEL_IN; k++)
    {
        for (i = 0; i < HEIGHT + 2; i++)
        {
            for (j = 0; j < WIDTH + 2; j++)
            {
                if (i == 0 || i == HEIGHT + 1 || j == 0 || j == WIDTH + 1)
                {
                    input_padding[index_padding] = 0;
                    index_padding += 1;
                }
                else
                {
                    input_padding[index_padding] = input[index];
                    index += 1;
                    index_padding += 1;
                }
                //  cout <<input_padding[index_padding - 1]<<" " << endl;
            }
        }
    }

    int c_out, c_in, h, w, c_i, c_o, hi;
    int index_in;
    int index_wt;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH; c_out += HEIGHT * WIDTH, c_o++)
    {
        for (h = 0, hi = 0;hi < HEIGHT; h += WIDTH, hi += 1)
        {
            for (w = 0; w < WIDTH; w++)
            {
#pragma HLS PIPELINE
                for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
#pragma HLS UNROLL
                    index_in = c_i * (HEIGHT + 2) * (WIDTH + 2) + h + w + WIDTH + 3 + hi * 2;
                    index_wt = c_o * CHANNEL_IN * 9 + c_i * 9;
                    result +=
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 3], weight[index_wt]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 2], weight[index_wt + 1]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 1], weight[index_wt + 2]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - 1], weight[index_wt + 3]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in], weight[index_wt + 4]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + 1], weight[index_wt + 5]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 1], weight[index_wt + 6]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 2], weight[index_wt + 7]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 3], weight[index_wt + 8]);
                }
  //              cout <<result<<" "<< endl;
                output[c_out + h + w] = result;
                result = 0;
            }
            //           cout << "\n" << endl;
        }
    }
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
void shift_stride2(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_INPUT *input_padding,
                   int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    FIX_OUTPUT result = 0;
    int i, j, k, index = 0, index_padding = 0;
    // padding
    for (k = 0; k < CHANNEL_IN; k++)
    {
        for (i = 0; i < HEIGHT + 2; i++)
        {
            for (j = 0; j < WIDTH + 2; j++)
            {
                if (i == 0 || i == HEIGHT + 1 || j == 0 || j == WIDTH + 1)
                {
                    input_padding[index_padding] = 0;
                    index_padding += 1;
                }
                else
                {
                    input_padding[index_padding] = input[index];
                    index += 1;
                    index_padding += 1;
                }
            }
        }
    }
    int c_out, c_in, h, w, c_i, c_o, hi;
    int index_in;
    int index_wt;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH / 4; c_out += HEIGHT * WIDTH / 4, c_o++)
    {
        for (h = 0, hi = 0; h < HEIGHT * WIDTH / 4; h += WIDTH / 2, hi += 1)
        {
            for (w = 0; w < WIDTH / 2; w += 1)
            {
#pragma HLS PIPELINE
                for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
#pragma HLS UNROLL
                    index_in = c_i * (HEIGHT + 2) * (WIDTH + 2) + 2 * w + WIDTH + 3 + hi * 2 * (WIDTH + 2);
                    index_wt = c_o * CHANNEL_IN * 9 + c_i * 9;
                    result +=
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 3], weight[index_wt]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 2], weight[index_wt + 1]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - WIDTH - 1], weight[index_wt + 2]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in - 1], weight[index_wt + 3]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in], weight[index_wt + 4]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + 1], weight[index_wt + 5]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 1], weight[index_wt + 6]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 2], weight[index_wt + 7]) +
                        shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input_padding[index_in + WIDTH + 3], weight[index_wt + 8]);
                }
                output[c_out + w + h] = result;
                result = 0;
            }
        }
    }
}

FIX_ACC abss(FIX_ACC input)
{
    if (input > 0)
        return input;
    else
        return (-input);
}

template <typename FIX_INPUT, typename FIX_ADD, typename FIX_OUTPUT>
void adder(FIX_INPUT *input, FIX_INPUT *input_padding, FIX_ADD *add, FIX_OUTPUT *output, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_OUT*HEIGHT*WIDTH]
    // add size is [CHANNEL_OUT*CHANNEL_OUT*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    FIX_MAX result = 0;
    //clear input_buf
    int i, j, k, index = 0, index_padding = 0;
    // padding
    for (k = 0; k < CHANNEL_OUT; k++)
    {
        for (i = 0; i < HEIGHT + 2; i++)
        {
            for (j = 0; j < WIDTH + 2; j++)
            {
                if (i == 0 || i == HEIGHT + 1 || j == 0 || j == WIDTH + 1)
                {
                    input_padding[index_padding] = 0;
                    index_padding += 1;
                }
                else
                {
                    input_padding[index_padding] = input[index];
                    index += 1;
                    index_padding += 1;
                }
            }
        }
    }
    int c_out, c_in, h, w, c_i, c_o, hi;
    int index_in;
    int index_wt;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH; c_out += HEIGHT * WIDTH, c_o++)
    {
        for (h = 0, hi = 0; h < HEIGHT * WIDTH; h += WIDTH, hi++)
        {
            for (w = 0; w < WIDTH; w++)
            {
#pragma HLS PIPELINE
                for (c_in = 0, c_i = 0; c_in < CHANNEL_OUT * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
#pragma HLS UNROLL
                    index_in = c_i * (HEIGHT + 2) * (WIDTH + 2) + h + w + WIDTH + 3 + hi * 2;
                    index_wt = c_o * CHANNEL_OUT * 9 + c_i * 9;
                    result -= abss(input_padding[index_in - WIDTH - 3] - add[index_wt]) +
                              abss(input_padding[index_in - WIDTH - 2] - add[index_wt + 1]) +
                              abss(input_padding[index_in - WIDTH - 1] - add[index_wt + 2]) +
                              abss(input_padding[index_in - 1] - add[index_wt + 3]) +
                              abss(input_padding[index_in] - add[index_wt + 4]) +
                              abss(input_padding[index_in + 1] - add[index_wt + 5]) +
                              abss(input_padding[index_in + WIDTH + 1] - add[index_wt + 6]) +
                              abss(input_padding[index_in + WIDTH + 2] - add[index_wt + 7]) +
                              abss(input_padding[index_in + WIDTH + 3] - add[index_wt + 8]);
                }
                output[c_out + h + w] = result / 32;
                result = 0;
            }
        }
    }
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_ADD, typename FIX_OUTPUT>
void conv3x3_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_ADD *add, FIX_OUTPUT *output, FIX_OUTPUT *result,
                     FIX_INPUT *input_padding, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    shift_stride1<FIX_INPUT, FIX_WEIGHT, FIX_OUTPUT>(input, weight, result, input_padding, CHANNEL_IN, CHANNEL_OUT, HEIGHT, WIDTH);
    adder<FIX_OUTPUT, FIX_ADD, FIX_OUTPUT>(result, input_padding, add, output, CHANNEL_OUT, HEIGHT, WIDTH);
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_ADD, typename FIX_OUTPUT>
void conv3x3_stride2(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_ADD *add, FIX_OUTPUT *output, FIX_OUTPUT *result,
                     FIX_INPUT *input_padding, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    shift_stride2<FIX_INPUT, FIX_WEIGHT, FIX_OUTPUT>(input, weight, result, input_padding, CHANNEL_IN, CHANNEL_OUT, HEIGHT, WIDTH);
    adder<FIX_OUTPUT, FIX_ADD, FIX_OUTPUT>(result, input_padding, add, output, CHANNEL_OUT, HEIGHT / 2, WIDTH / 2);
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT, typename FIX_BIAS>
void batchnorm(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_BIAS *bias,
               int CHANNEL_IN, int HEIGHT, int WIDTH)
{
    int c_in, h, w, i, c_i;
    int index;
    for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
    {
#pragma HLS PIPELINE
        for (i = 0; i < HEIGHT * WIDTH; i++)
        {
#pragma HLS UNROLL
            output[c_in + i] = input[c_in + i] * weight[c_i] + bias[c_i];
        }
    }
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT, typename FIX_ADD>
void downsample(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *res, FIX_OUTPUT *output, FIX_ADD *add,
                int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
#pragma HLS RESOURCE variable = return core = AddSub_DSP
    int c_in, h, w, i, c_out, c_o, c_i, hi;
    int index_out, index_wt, index_in;
    FIX_MAX result = 0;

    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH / 4; c_out += HEIGHT * WIDTH / 4, c_o++)
    {
        for (h = 0, hi = 0; h < HEIGHT * WIDTH / 4; h += WIDTH / 2, hi += 1)
        {
            for (w = 0; w < WIDTH / 2; w += 1)
            {
#pragma HLS PIPELINE
                for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
#pragma HLS UNROLL
                    index_in = c_in + 2 * hi * WIDTH + w * 2;
                    index_wt = c_o * CHANNEL_IN + c_i;
                    result += shift<FIX_INPUT, FIX_WEIGHT,FIX_OUTPUT>(input[index_in],weight[index_wt]);
                }
                res[c_out + w + h] = result;
                result = 0;
            }
        }
    }

    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH / 4; c_out += HEIGHT * WIDTH / 4, c_o++)
    {
        for (h = 0, hi = 0; h < HEIGHT * WIDTH / 4; h += WIDTH / 2, hi++)
        {
            for (w = 0; w < WIDTH / 2; w++)
            {
#pragma HLS PIPELINE
                for (c_in = 0, c_i = 0; c_in < CHANNEL_OUT * HEIGHT * WIDTH / 4; c_in += HEIGHT * WIDTH / 4, c_i++)
                {
#pragma HLS UNROLL
                    index_in = c_in + h + w;
                    index_wt = c_o * CHANNEL_OUT + c_i;
                    result -= abss(res[index_in] - add[index_wt]);
                }
                output[c_out + h + w] = result / 32;
                result = 0;
            }
        }
    }
}

template <typename FIX_INPUT, typename FIX_OUTPUT>
void relu(FIX_INPUT *input, FIX_OUTPUT *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (input[i] < 0)
        {
            output[i] = 0;
        }
        else
            output[i] = input[i];
    }
}

template <typename FIX_INPUT, typename FIX_OUTPUT>
void addd(FIX_INPUT *input1, FIX_INPUT *input2, FIX_OUTPUT *output, int size)
{
    for (int i = 0; i < size; i++)
    {
#pragma HLS UNROLL
        output[i] = input1[i] + input2[i];
    }
}

void avgpool(FIX_ACC *input, FIX_ACC output[64])
{
    int j;
    FIX_ACC sum = 0;
    for (int i = 0; i < 64 * 64; i += 64)
    {
        for (j = 0; j < 64; j++)
        {
            sum += input[i + j];
        }
        output[i / 64] = sum / 64;
        sum = 0;
    }
}

void fc(FIX_ACC input[64], FIX_WT_4 weight[10 * 64], FIX_ACC output[10])
{
    int j;
    FIX_ACC sum = 0;
    for (int i = 0; i < 640; i += 64)
    {
        for (j = 0; j < 64; j++)
        {
            sum += weight[i + j] * input[j];
        }
        output[i] = sum;
        sum = 0;
    }
}

void find_max(int r[1], FIX_ACC input[10])
{
    int index;
    int max_index = 0;
    FIX_ACC res1;
    FIX_ACC res2;
    for (index = 0; index < 10; index++)
    {
        res1 = input[index];
        res2 = input[max_index];
        if (res1 > res2)
        {
            max_index = index;
        }
    }
    r[0] = max_index;
}

void shiftadder(FIX_FM input[3 * 32 * 32], FIX_WT_4 weight[74160], FIX_ADD_10 add[96768], FIX_BN_WT bn_wt[336],
                FIX_BN_BIAS bn_bias[336], int r[1])
{
#pragma HLS INTERFACE m_axi depth = 3072 port = input bundle = DATA
#pragma HLS INTERFACE m_axi depth = 74160 port = weight bundle = DATA
#pragma HLS INTERFACE m_axi depth = 96768 port = add bundle = DATA
#pragma HLS INTERFACE m_axi depth = 336 port = bn_wt bundle = DATA
#pragma HLS INTERFACE m_axi depth = 336 port = bn_bias bundle = DATA
#pragma HLS INTERFACE m_axi depth = 1 port = r bundle = DATA

#pragma HLS INTERFACE s_axilite register port = input bundle = CTRL
#pragma HLS INTERFACE s_axilite register port = weight bundle = CTRL
#pragma HLS INTERFACE s_axilite register port = add bundle = CTRL
#pragma HLS INTERFACE s_axilite register port = bn_wt bundle = CTRL
#pragma HLS INTERFACE s_axilite register port = bn_bias bundle = CTRL
#pragma HLS INTERFACE s_axilite register port = r bundle = CTRL

#pragma HLS INTERFACE s_axilite register port = return bundle = CTRL

    FIX_ACC output1[16 * 32 * 32];
    FIX_ACC output2[16 * 32 * 32];
    FIX_ACC output3[16 * 32 * 32];
    FIX_ACC output4[16 * 32 * 32];
    FIX_ACC output[64];
    FIX_ACC out[10];
    FIX_ACC result[16 * 32 * 32];
    FIX_FM fm_padding[3 * 34 * 34];
    FIX_ACC padding[16 * 34 * 34];
    // first step
    shift_stride1<FIX_FM, FIX_WT_4, FIX_ACC>(input, weight, output1, fm_padding, 3, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC,FIX_BN_BIAS>(output1, bn_wt, output2, bn_bias, 16, 32, 32);
    // layer1 start
    // output2 is the start to be saved becuase of short cut
    // block1
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 16 * 32 * 32);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 16 * 32 * 32);
    // block2
  /*  conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 16 * 32 * 32);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 16 * 32 * 32);
    // block3
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 16, 16, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 16, 32, 32);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 16 * 32 * 32);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 16 * 32 * 32); */
    // layer1 end

    // layer2 start
    // output2 is the start to be saved becuase of short cut
    //block 1
    conv3x3_stride2<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 16, 32, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 32, 32, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    downsample<FIX_ACC, FIX_WT_4, FIX_ACC, FIX_ADD_10>(output2, weight, result, output4, add, 16, 32, 32, 32);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output4, bn_wt, output2, bn_bias, 32, 16, 16);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 32 * 16 * 16);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 32 * 16 * 16);
    // block 2
/*    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 32, 32, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 32, 32, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 32 * 16 * 16);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 32 * 16 * 16);
    //block 3
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 32, 32, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 32, 32, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 32, 16, 16);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 32 * 16 * 16);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 32 * 16 * 16);*/
    //layer2 end

    // layer3 start
    //output2 is the start to be saved becuase of short cut
    // block1
    conv3x3_stride2<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 32, 64, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 64, 64, 8, 8);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    downsample<FIX_ACC, FIX_WT_4, FIX_ACC, FIX_ADD_10>(output2, weight, result, output4, add, 32, 64, 16, 16);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output4, bn_wt, output2, bn_bias, 64, 8, 8);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 64 * 8 * 8);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 64 * 8 * 8);
    //block2
 /*   conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 64, 64, 8, 8);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 64, 64, 8, 8);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 64 * 8 * 8);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 64 * 8 * 8);
    //block3
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output2, weight, add, output1, result, padding, 64, 64, 8, 8);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    conv3x3_stride1<FIX_ACC, FIX_WT_4, FIX_ADD_10, FIX_ACC>(output3, weight, add, output1, result, padding, 64, 64, 8, 8);
    batchnorm<FIX_ACC, FIX_BN_WT, FIX_ACC, FIX_BN_BIAS>(output1, bn_wt, output3, bn_bias, 64, 8, 8);
    addd<FIX_ACC, FIX_ACC>(output2, output3, output1, 64 * 8 * 8);
    relu<FIX_ACC, FIX_ACC>(output1, output2, 64 * 8 * 8); */
    // layer3 end
    avgpool(output2, output);
    fc(output, weight, out);
    find_max(r, out);
}
