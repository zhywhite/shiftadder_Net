#include "shiftadder.h"

template<typename FIX_INPUT, typename FIX_WEIGHT,typename FIX_OUTPUT,typename FIX_BIAS>
void batchnorm(FIX_INPUT *input, FIX_WEIGHT *weight,FIX_OUTPUT *output,FIX_BIAS *bias,
                int CHANNEL_IN,int HEIGHT, int WIDTH){
    int c_in, h, w, i;
    int index;
    for (c_in = 0; c_in < CHANNEL_IN*HEIGHT*WIDTH; c_in+= HEIGHT*WIDTH)
    {
        for ( i = 0; i < HEIGHT*WIDTH; i++)
        {
            output[c_in + i] = input[c_in + i] >> weight[c_in] + bias[c_in];
        }
    }
}

template<typename FIX_INPUT, typename FIX_WEIGHT,typename FIX_OUTPUT>
FIX_OUTPUT shift(FIX_INPUT input, FIX_WEIGHT weight){
    FIX_IN_4 wt = 7;
    if (weight==0)
    {
        return 0;
    }
    else {
        wt = wt & weight;
        if (weight > 0)
        {
            return(input >> wt);
        }
        else return (-(input >> wt))
    }
}

template<typename FIX_INPUT, typename FIX_WEIGHT,typename FIX_OUTPUT,typename FIX_ADD>
void downsample(FIX_INPUT *input,FIX_INPUT *input_padding, FIX_WEIGHT *weight,FIX_OUTPUT *output,FIX_ADD *add, 
                int CHANNEL_IN,int CHANNEL_OUT,int HEIGHT, int WIDTH){
    int c_in, h, w, i,c_out,c_o,c_i;
    int index_out,index_wt,index_in;
    FIX_OUTPUT result;
    for (c_out = 0, c_o = 0; c_out < CHANNEL_OUT * HEIGHT * WIDTH /4; c_out += HEIGHT * WIDTH/4, c_o++)
    {
        for (h = 0; h < HEIGHT * WIDTH; h += 2*WIDTH)
        {
            for (w = 0; w < WIDTH; w += 2)
            {
                for (c_in = 0, c_i = 0; c_in < CHANNEL_IN * HEIGHT * WIDTH; c_in += HEIGHT * WIDTH, c_i++)
                {
                    index_in = c_in + h + w;
                    index_wt = c_out + c_i * HEIGHT  * WIDTH / 4 + h/2 + w/2;
                    result += input_padding[index_in] >> weight[index_wt];
                }
                output[index_out + w/2] = result;
                result = 0;
            }
            index_out += WIDTH/2;
        }
        index_out += HEIGHT/2 * WIDTH/2;
    }
    adder(result,input_padding, add, output, CHANNEL_OUT, HEIGHT/2, WIDTH/2);
}

template<typename FIX_INPUT,typename FIX_OUTPUT>
void relu(FIX_INPUT *input, FIX_OUTPUT *output,int size){
    for (int i = 0; i < size; i++)
    {
        if (input [i] < 0)
        {
            output[i] = 0;
        }
        else output[i] = input[i];
    }
}
