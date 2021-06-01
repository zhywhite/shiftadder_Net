#include "shiftadder.h"
#include "stdlib.h"
using namespace std;

int main(){
	FIX_WT_4 weight[74160];
	FIX_BN_WT bn_wt[336];

	FIX_ADD_10 add[96768];
	FIX_BN_BIAS bn_bias[336];

	FIX_FM input[3*1024];
	int r[1];
	for(long i_2 = 0; i_2<74160; i_2++){
	//	cout <<"weight" <<weights[i_2] << "\n" << endl;
	    weight[i_2] = -1;
	}

	for(int i_1 = 0; i_1<336; i_1++){
	    bn_wt[i_1] = 1;
	    bn_bias[i_1] = 1;
	}
	for(int i_1 = 0; i_1<96768; i_1++){
		add[i_1] = 1;
	}
	for(int i_1 = 0; i_1<3*1024; i_1++){
		input[i_1] = 1;
	}
	for(int i_1 = 0; i_1<640; i_1++){
		input[i_1] = 1;
	}
    shiftadder(input,weight,add,bn_wt,bn_bias,r);
	cout<<"res:"<<r[0]<<"\n" << endl;
	return 0;
}
