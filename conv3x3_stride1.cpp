#include "shiftadder.h"

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_ADD, typename FIX_OUTPUT>
void conv3x3_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_ADD *add, FIX_OUTPUT *output, FIX_OUTPUT *result,
                     FIX_INPUT *input_padding, int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    shift_stride1<FIX_INPUT, FIX_WEIGHT, FIX_OUTPUT>(input, weight, result,input_padding,CHANNEL_IN, CHANNEL_OUT, HEIGHT, WIDTH);
    adder<FIX_OUTPUT, FIX_ADD, FIX_OUTPUT>(result, input_padding, add, output, CHANNEL_OUT, HEIGHT, WIDTH);
}

template <typename FIX_INPUT, typename FIX_WEIGHT, typename FIX_OUTPUT>
void shift_stride1(FIX_INPUT *input, FIX_WEIGHT *weight, FIX_OUTPUT *output, FIX_INPUT *input_padding,
                   int CHANNEL_IN, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_IN*HEIGHT*WIDTH]
    // weight size is [CHANNEL_OUT*CHANNEL_IN*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    FIX_OUTPUT result;
    //clear input_buf
    int i, j, k, index = 0, index_padding = 0;
    // padding
    for (k = 0; k < CHANNEL_IN; k++)
    {
        for (i = 0; i < HEIGHT + 1; i++)
        {
            for (j = 0; j < WIDTH + 1; j++)
            {
                if (i == 0 || i == HEIGHT || j == 0 || j == WIDTH)
                {
                    input_padding[index] = 0;
                    index += 1;
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
    int c_out, c_in, h, w, c_i, c_o;
    int index_in;
    int index_wt;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH; c_out += HEIGHT * WIDTH, c_o++)
    {
        for (h = 0; h < HEIGHT * WIDTH; h += WIDTH)
        {
            for (w = 0; w < WIDTH; w++)
            {
                for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
                    index_in = c_in + h + w;
                    index_wt = c_o * CHANNEL_IN + c_i * 9;
                    result +=
                        input_padding[index_in - WIDTH - 1] >> weight[index_wt] +
                                                                   input_padding[index_in - WIDTH] >>
                        weight[index_wt + 1] +
                            input_padding[index_in - WIDTH + 1] >>
                        weight[index_wt + 2] +
                            input_padding[index_in - 1] >>
                        weight[index_wt + 3] +
                            input_padding[index_in] >>
                        weight[index_wt + 4] +
                            input_padding[index_in + 1] >>
                        weight[index_wt + 5] +
                            input_padding[index_in + WIDTH - 1] >>
                        weight[index_wt + 6] +
                            input_padding[index_in + WIDTH] >>
                        weight[index_wt + 7] +
                            input_padding[index_in + WIDTH + 1] >>
                        weight[index_wt + 8];
                }
                output[c_out + h + w] = result;
                result = 0;
            }
        }
    }
}

template <typename FIX_INPUT, typename FIX_ADD, typename FIX_OUTPUT>
void adder(FIX_INPUT *input,FIX_INPUT *input_padding, FIX_ADD *add, FIX_OUTPUT *output, int CHANNEL_OUT, int HEIGHT, int WIDTH)
{
    // input size is [CHANNEL_OUT*HEIGHT*WIDTH]
    // add size is [CHANNEL_OUT*CHANNEL_OUT*3*3]
    // output size is [CHNNEL_OUT*HEIGHT*WIDTH]
    FIX_OUTPUT result;
    //clear input_buf
    int i, j, k, index = 0, index_padding = 0;
    // padding
    for (k = 0; k < CHANNEL_OUT; k++)
    {
        for (i = 0; i < HEIGHT + 1; i++)
        {
            for (j = 0; j < WIDTH + 1; j++)
            {
                if (i == 0 || i == HEIGHT || j == 0 || j == WIDTH)
                {
                    input_padding[index] = 0;
                    index += 1;
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
    int c_out, c_in, h, w, c_i, c_o;
    int index_in;
    int index_wt;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH; c_out += HEIGHT * WIDTH, c_o++)
    {
        for (h = 0; h < HEIGHT * WIDTH; h += WIDTH)
        {
            for (w = 0; w < WIDTH; w++)
            {
                for (c_in = 0, c_i = 0; c_in < CHANNEL_OUT * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
                    index_in = c_in + h + w;
                    index_wt = c_o * CHANNEL_OUT + c_i * 9;
                    result -= abs(input_padding[index_in - WIDTH - 1] - add[index_wt]) +
                              abs(input_padding[index_in - WIDTH] - add[index_wt + 1]) +
                              abs(input_padding[index_in - WIDTH + 1] - add[index_wt + 2]) +
                              abs(input_padding[index_in - 1] - add[index_wt + 3]) +
                              abs(input_padding[index_in] - add[index_wt + 4]) +
                              abs(input_padding[index_in + 1] - add[index_wt + 5]) +
                              abs(input_padding[index_in + WIDTH - 1] - add[index_wt + 6]) +
                              abs(input_padding[index_in + WIDTH] - add[index_wt + 7]) +
                              abs(input_padding[index_in + WIDTH + 1] - add[index_wt + 8]);
                }
                output[c_out + h + w] = result;
                result = 0;
            }
        }
    }
}
