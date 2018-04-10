#ifndef _DEFINE_H
#define _DEFINE_H
#include <stdlib.h>
#include <stdio.h>

#define TESTING //comment this for synthesis
#define SILENT  //un-comment for synthesis

//#define LINUX
//#define TEST_ADAPT //do not uncomment

 //#define THREAD_N 4

#ifdef TESTING
	#include <math.h>
	typedef double data_type;

#else
	#include <math.h>
	//#include <ap_fixed.h>
	//#include <hls_stream.h>
	//#include <ap_int.h>
	//typedef ap_fixed<32, 32, AP_RND, AP_SAT> data_type;
	typedef double data_type;
#endif

//#include <hls_math.h>

#define VAR_THRES 0.001

//typedef ap_int<9> data_type;


// Adaptation
#define ALPHA 0.01
#define	LIST_DEPTH 65 //mixture depth.. During Adaptation
#define M_PI  3.14159265358979323846

typedef int t_int;

//typedef ap_uint<16> t_int;

#define DBL_MAX 1e06
#define DBL_MIN 1e-06

#define FLT_EPSILON 1e-10
#define KMEANS_MAX_ERR 1e-4
#define MAX_COMPONENTS 5 //applies only in KMEANS during FIT
#define INPUT_DEPTH 100
#define LAMBDA0 5

#define BETA 1e6
#define CE 1e-5

#define CRIT 1e-6

#define KMEANS_DIM 1

//#define SHOW(a) std::cout << #a << ": " << (a) << std::endl
#endif

