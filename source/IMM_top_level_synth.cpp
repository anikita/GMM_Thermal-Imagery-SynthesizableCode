#include <stdio.h>
#include <stdlib.h> 
#include "IMM.h"

#include "IMM_Adaptation.h"

using namespace std;

data_type IMM_top_level_synth(void){

	data_type input0[INPUT_DEPTH] = {14.151220496789,47.454596310295,53.961569152495,49.192148348788,16.475781691182,16.269967285869,14.737190467031,50.898556286678,16.079905635544,15.137034179200,14.950776748154,14.555217336074,47.586226431276,15.124072299443,51.204690856086,15.173560277214,18.729110023423,53.854118147522,52.164519573341,14.660659766189,16.342740750267,48.258377449684,14.974085893929,19.539873183333,17.112079917329,53.750329857351,18.258844403380,52.078925788275,55.242069743264,49.329963142527,16.911747848911,48.586085630130,46.166582295940,48.461461782806,14.625934519030,48.031702294963,17.864795292577,15.184636966442,51.860620859623,49.506505029874,15.575046452378,49.954183109161,18.482401919649,16.800377221905,53.289159878769,17.533438128121,51.093074164144,14.798005314404,15.727894900857,47.896205831717,46.836375193986,14.974658499294,16.754355851261,16.813126592968,18.811973656365,16.125639009249,16.319821612765,16.133130475571,15.645815325361,47.959257430490,14.794651735559,15.039496089307,47.715277338648,16.292914471412,52.471429799794,14.587400880866,51.196251473859,15.808955110960,50.165330269878,14.654312574415,51.829477886651,48.309936681925,51.855229997897,51.195168288521,13.848392654607,16.077345876106,15.782352145681,48.163254516647,51.232584188683,19.420798462542,50.587200856152,50.211489693907,14.341445477764,16.142324872446,49.118166116332,18.529415726448,50.551490359777,15.981646587388,47.796856091868,15.032246637422,14.250804226909,50.545898935466,51.342839410461,47.700052578218,15.589066388948,48.015648330821,16.425950238432,50.488325849342,51.891123127578,50.772895443838};

	data_type a;
	data_type b=32.333;

 

/*
	data_type my_mixtures1[3][MAX_COMPONENTS];
	data_type sample_param1[4];
	int no_components1=1;

	data_type my_mixtures2[3][MAX_COMPONENTS];
	data_type sample_param2[4];
	int no_components2=1;

	data_type my_mixtures3[3][MAX_COMPONENTS];
	data_type sample_param3[4];
	int no_components3=1;

	data_type my_mixtures4[3][MAX_COMPONENTS];
	data_type sample_param4[4];
	int no_components4=2;

	data_type my_mixtures5[3][MAX_COMPONENTS];
	data_type sample_param5[4];
	int no_components5=1;

	data_type my_mixtures6[3][MAX_COMPONENTS];
	data_type sample_param6[4];
	int no_components6=1;

	data_type my_mixtures7[3][MAX_COMPONENTS];
	data_type sample_param7[4];
	int no_components7=1;

	data_type my_mixtures8[3][MAX_COMPONENTS];
	data_type sample_param8[4];
	int no_components8=2;

	data_type my_mixtures9[3][MAX_COMPONENTS];
	data_type sample_param9[4];
	int no_components9=1;

	data_type my_mixtures10[3][MAX_COMPONENTS];
	data_type sample_param10[4];
	int no_components10=10;

	data_type my_mixtures11[3][MAX_COMPONENTS];
	data_type sample_param11[4];
	int no_components11=1;

	data_type my_mixtures12[3][MAX_COMPONENTS];
	data_type sample_param12[4];
	int no_components12=1;

	data_type my_mixtures13[3][MAX_COMPONENTS];
	data_type sample_param13[4];
	int no_components13=1;

	data_type my_mixtures14[3][MAX_COMPONENTS];
	data_type sample_param14[4];
	int no_components14=1;

	data_type my_mixtures15[3][MAX_COMPONENTS];
	data_type sample_param15[4];
	int no_components15=1;
*/

//	IMM imm00(input0, sample_param00);
//	imm00.Fit(my_mixtures00, no_components00);

	data_type my_mixtures0[3][LIST_DEPTH];
	data_type sample_param0[4];
	int no_components0=2;

	my_mixtures0[0][0]=0.53; //pi
	my_mixtures0[1][0]=16.04;//mu
	my_mixtures0[2][0]=0.192; //std

	my_mixtures0[0][1]=0.47; //pi
	my_mixtures0[1][1]=50.15;//mu
	my_mixtures0[2][1]=0.305; //std


	sample_param0[0]=17.022;
	sample_param0[1]=32.077;
	sample_param0[2]=132220.091;


	IMM_Adaptation imm_adapt0(0);

	a=  imm_adapt0.Adaptation(3, 50.772895443838, my_mixtures0, no_components0, sample_param0);
	a=no_components0;

	/*
	IMM_Adaptation imm_adapt1(0);
	a=  imm_adapt1.Adaptation(26, 50.772895443838, my_mixtures1, no_components1, sample_param1);

	IMM_Adaptation imm_adapt2(0);
	a=  imm_adapt2.Adaptation(27, 50.772895443838, my_mixtures2, no_components2, sample_param2);

	IMM_Adaptation imm_adapt3(0);
	a=  imm_adapt3.Adaptation(23, 50.772895443838, my_mixtures3, no_components3, sample_param3);

	IMM_Adaptation imm_adapt4(0);
	a=  imm_adapt4.Adaptation(25, 50.772895443838, my_mixtures4, no_components4, sample_param4);

	IMM_Adaptation imm_adapt5(0);
	a=  imm_adapt5.Adaptation(26, 50.772895443838, my_mixtures5, no_components5, sample_param5);

	IMM_Adaptation imm_adapt6(0);
	a=  imm_adapt6.Adaptation(27, 50.772895443838, my_mixtures6, no_components6, sample_param6);

	IMM_Adaptation imm_adapt7(0);
	a=  imm_adapt7.Adaptation(23, 50.772895443838, my_mixtures7, no_components7, sample_param7);

	IMM_Adaptation imm_adapt8(0);
	a=  imm_adapt8.Adaptation(25, 50.772895443838, my_mixtures8, no_components8, sample_param8);

	IMM_Adaptation imm_adapt9(0);
	a=  imm_adapt9.Adaptation(26, 50.772895443838, my_mixtures9, no_components9, sample_param9);

	IMM_Adaptation imm_adapt10(0);
	a=  imm_adapt10.Adaptation(27, 50.772895443838, my_mixtures10, no_components10, sample_param10);

	IMM_Adaptation imm_adapt11(0);
	a=  imm_adapt11.Adaptation(23, 50.772895443838, my_mixtures11, no_components11, sample_param11);

	IMM_Adaptation imm_adapt12(0);
	a=  imm_adapt12.Adaptation(27, 50.772895443838, my_mixtures12, no_components12, sample_param12);

	IMM_Adaptation imm_adapt13(0);
	a=  imm_adapt13.Adaptation(23, 50.772895443838, my_mixtures13, no_components13, sample_param13);

	IMM_Adaptation imm_adapt14(0);
	a=  imm_adapt14.Adaptation(25, 50.772895443838, my_mixtures14, no_components14, sample_param14);

	IMM_Adaptation imm_adapt15(0);
	a=  imm_adapt15.Adaptation(26, 50.772895443838, my_mixtures15, no_components15, sample_param15);

	*/




//	imm1.Fit(my_mixtures, no_components);
//	a= imm1.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);


	/*
	IMM imm2(input0, sample_param);
	imm2.Fit(my_mixtures, no_components);
	a= imm2.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);

	IMM imm3(input0, sample_param);
	imm3.Fit(my_mixtures, no_components);
	a= imm3.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);


	IMM imm4(input0, sample_param);
	imm4.Fit(my_mixtures, no_components);
	a= imm4.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);

	IMM imm5(input0, sample_param);
	imm5.Fit(my_mixtures, no_components);
	a= imm5.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);

	IMM imm6(input0, sample_param);
	imm6.Fit(my_mixtures, no_components);
	a= imm6.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);

	IMM imm7(input0, sample_param);
	imm7.Fit(my_mixtures, no_components);
	a= imm7.Adaptation(25, 50.772895443838, my_mixtures, no_components, sample_param);
	*/
	return a;

}
