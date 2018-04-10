#ifndef _IMM_H_
#define _IMM_H_
//#include <stdlib.h>
#include "define.h"

class IMM
{

public:
	//function prototypes
	 IMM(data_type input_[INPUT_DEPTH],  data_type (&sample_param)[4] );
	 void Init(data_type input_[INPUT_DEPTH],  data_type (&sample_param)[4] );
	 void Fit(data_type (&mixtures)[3][LIST_DEPTH],int &number_of_components);
	 data_type Adaptation(data_type x_new, data_type x_remove, data_type (&mixtures_pass)[3][LIST_DEPTH],int &no_component_pass, data_type (&sample_param)[4] );
	 data_type uniform_pdf(data_type x, data_type mu, data_type sigma);
	 data_type norm_pdf(data_type x, data_type mu, data_type sigma);
	 void Estep(void);
	 void Mstep(void);

	 void kmeans(data_type data[INPUT_DEPTH]);
	 data_type digamma(data_type x);
	 void show(data_type x[MAX_COMPONENTS], char str[12]);
	 void show2(data_type x[INPUT_DEPTH][MAX_COMPONENTS], char str[12]);
	 data_type input[INPUT_DEPTH];
	 data_type input_squared[INPUT_DEPTH];
private:
	//Adaptation

	//data_type mixtures[3][LIST_DEPTH];

	//kmeans
	data_type centroids[MAX_COMPONENTS][KMEANS_DIM];
	t_int clusters_size[MAX_COMPONENTS];
	data_type inertia[MAX_COMPONENTS][KMEANS_DIM];
	int	  labels[INPUT_DEPTH]; // labels for each input

	//constructor body
	data_type Pi[MAX_COMPONENTS];
	data_type MuS[MAX_COMPONENTS];
	data_type mS[MAX_COMPONENTS];
	data_type vS[MAX_COMPONENTS];
	data_type cS[MAX_COMPONENTS];
	data_type m0;
	data_type v0; 
	data_type bS[MAX_COMPONENTS];
	data_type LambdaS[MAX_COMPONENTS];
	data_type sum_LambdaS;

	//Estep
	data_type log_pi_s_hat[MAX_COMPONENTS];
	data_type log_beta_s_hat[MAX_COMPONENTS];
	data_type beta_s_mean[MAX_COMPONENTS];
	data_type gamma_n_s_hat[INPUT_DEPTH][MAX_COMPONENTS];
	data_type gamma_n_s[INPUT_DEPTH][MAX_COMPONENTS];


	//Mstep
	data_type pi_s_mean[MAX_COMPONENTS];
	data_type N_s_mean[MAX_COMPONENTS];
	data_type y_s_mean[MAX_COMPONENTS];
	data_type y_s_square_mean[MAX_COMPONENTS];
	data_type sigma_s_square_mean[MAX_COMPONENTS];
	data_type LambdaS_[MAX_COMPONENTS];
	data_type one_div_bS [MAX_COMPONENTS];
	//data_type bS_[MAX_COMPONENTS];
	//data_type cS_[MAX_COMPONENTS];
	data_type m_data_s[MAX_COMPONENTS];
	data_type tau_data_s[MAX_COMPONENTS]; 
	data_type tau_s[MAX_COMPONENTS]; 
	//data_type mS_[MAX_COMPONENTS];
	//data_type vS_[MAX_COMPONENTS];

	//Fit
	data_type sigma_est[MAX_COMPONENTS];
	bool components[MAX_COMPONENTS];
	int	 number_of_components;
	data_type pi_est[MAX_COMPONENTS];
	data_type pis[MAX_COMPONENTS];
	
	//data_type mixture[3][MAX_COMPONENTS]; //will pass to adaptation


};

#endif
