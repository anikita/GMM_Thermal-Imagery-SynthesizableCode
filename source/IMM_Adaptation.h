#ifndef _IMM_ADAPT_H_
#define _IMM_ADAPT_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "define.h"
/*
#define MAX_COMPONENTS 5

typedef float data_type;
//data_type mixtures[3][MAX_COMPONENTS];

#define TESTING
#define	LIST_DEPTH 16
#define INPUT_DEPTH 100
#define DBL_MAX 1e06
#define DBL_MIN 1e-06
#define ALPHA 0.01

#define M_PI  3.14159265358979323846
*/

class IMM_Adaptation
{

public:

IMM_Adaptation(int a);
data_type Adaptation(data_type x_new, data_type x_remove, data_type (&mixtures_pass)[3][LIST_DEPTH],int &no_component_pass, data_type (&sample_param)[4] );
data_type uniform_pdf(data_type x, data_type mu, data_type sigma);
data_type norm_pdf(data_type x, data_type mu, data_type sigma);
};

#endif