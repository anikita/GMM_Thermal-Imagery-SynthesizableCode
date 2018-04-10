
//#include <float.h>
//#include <iostream>

//#include <math.h>
//#include <cmath>


#include "IMM.h"

//using namespace  ap_[u]fixed
//using namespace hls;
 IMM::IMM (data_type input_[INPUT_DEPTH], data_type (&sample_param)[4] ){


#ifdef TEST_ADAPT
number_of_components=2;

sample_param[0]=32.077;
sample_param[1]=17.022,
sample_param[2]=132220.109;

data_type mixtures_[3][2]=
{
	{0.470000,   0.530000}, //pis
	{50.151462, 16.043631}, //mean
	{0.305811,  0.192676}  //sigmas
};

for (int k=0;k<number_of_components;k++) for (int i=0;i<3;i++) mixtures[i][k]= mixtures_[i][k];

#endif
#ifndef TEST_ADAPT

	for (int x=0; x<INPUT_DEPTH; x++) {
 
		 input[x]=input_[x];
		 input_squared[x]=input_[x]*input_[x];
	}

	Init(input, sample_param);
#endif// end TEST_ADAPT

}

void IMM::Init (data_type input[INPUT_DEPTH], data_type (&sample_param)[4] ){

	 	//for (int x=0; x<INPUT_DEPTH; x++) input[x]=input_[x];

	////////////////////////////////////////////////// Sample Probability distribution computation  /////////////////////////
	data_type sample_sum=0;
	data_type std_sum=0;
	data_type sample_squared_sum =0;

	for (int x=0; x<INPUT_DEPTH; x++)  {
		sample_sum += input[x];
		sample_squared_sum += pow(input[x],2);
	}
	data_type sample_mean=sample_sum/(INPUT_DEPTH);
	for (int x=0; x<INPUT_DEPTH-1; x++) std_sum+= pow( input[x]- sample_mean ,2);

	data_type sample_sigma= sqrt(std_sum/(INPUT_DEPTH));

	sample_param[0]=sample_mean;
	sample_param[1]=sample_sigma;
	sample_param[2]=sample_squared_sum;
#ifndef SILENT
	printf("\nInitial sample sigma=%7.3f, mean=%7.3f, \nsample_squared_sum=%7.3f \n",sample_sigma, sample_mean,sample_squared_sum);
#endif
	//data_type sigma_alt = sqrt( sample_squared_sum/INPUT_DEPTH - pow(sample_mean,2 ) );
	//printf("first sample alt. sigma=%3.4f\n",sigma_alt);

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	 kmeans(input); //run kmenas
	 //getchar();
	 //for (int i=0; i<MAX_COMPONENTS; i++) printf("centdroids[%d](x,y)= %2.1f ,%2.1f\n", i, centroids[i][0],centroids[i][1]);
	 //for (int i=0; i<INPUT_DEPTH; i++) printf("labels[%d]= %d, data=%3.4f \n", i,labels[i], input[i]);
	 //for (int k=0; k<MAX_COMPONENTS; k++) printf("centroids[%d]= %3.4f, inertia[]=%3.4f, clusters_size=%d \n", k,centroids[k][0],inertia[k][0],clusters_size[k]);
	 //for (int i=0;i<INPUT_DEPTH;i++) labels[i]+=1;
	
	 //Pi
		//search labels and skip out the zero labels
		// normazilze 
	 /*
data_type mypi[MAX_COMPONENTS]={
7.800000000000000266e-01,
8.999999999999999667e-02,
4.000000000000000083e-02,
2.000000000000000042e-02,
7.000000000000000666e-02
};
*/
	int count=0;
	//data_type temp_pi[MAX_COMPONENTS]={0.010000000000,0.090000000000,0.120000000000,0.060000000000,0.030000000000,0.180000000000,0.040000000000,0.030000000000,0.010000000000,0.070000000000,0.030000000000,0.020000000000,0.040000000000,0.060000000000,0.080000000000,0.030000000000,0.010000000000,0.010000000000,0.060000000000,0.020000000000};
	for (int k=0; k<MAX_COMPONENTS; k++){ //for each component count the occurences in the data

		for (int i=0; i<INPUT_DEPTH; i++) if (labels[i]>0 && labels[i]==k) count++; //for each data potint count only if current cluster is found
		Pi[k]= (data_type) count/INPUT_DEPTH; //store the occurences normalized 
	
		//Pi[k]=mypi[k];//------------*************TESTING ONLY TEMPORARY

		count=0;
	}	

//show(Pi,"Pi");

	//Mus,mS
	//data_type mus_temp[MAX_COMPONENTS]={52.299023486232,17.393297969200,49.811032506559,15.591096870757,55.327271064505,16.432137583495,48.121898673166,19.670649033663,13.647031377703,51.602581826006,53.596737395866,47.111590564255,49.040245832087,17.981691885689,50.682678606352,52.705021061528,14.740167577302,12.639079445109,15.171509230405,14.308610349086};
	for (int k=0; k<MAX_COMPONENTS; k++) MuS[k]=mS[k]= centroids[k][0];  //mus_temp[k]; //TEMPORARY: 

	//vS, cS
	//data_type vS_temp[MAX_COMPONENTS]={0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880,0.145477434880};
	data_type sum_inertia=0;
	for (int k=0; k<MAX_COMPONENTS; k++)  sum_inertia+=inertia[k][0];
	// printf("sum_inertia=%3.4f\n",sum_inertia);
	//for (int k=0; k<MAX_COMPONENTS; k++)  cS[k]=vS[k]=  pow(sum_inertia,2)/(INPUT_DEPTH); //vS_temp[k]; //TEMPORARY:

	data_type var_c_norm[MAX_COMPONENTS];

	for (int k=0; k<MAX_COMPONENTS; k++) {
		data_type temp_data = inertia[k][0] / (Pi[k]*INPUT_DEPTH);   
		if (temp_data/ (Pi[k]*INPUT_DEPTH) > VAR_THRES) cS[k]=vS[k]= var_c_norm[k]=temp_data/ (Pi[k]*INPUT_DEPTH);
		else cS[k]=vS[k]= var_c_norm[k]=VAR_THRES;
	}

 //show(var_c_norm,"var_c_norm");

	//m0
	m0=0; data_type input_max=DBL_MIN; data_type input_min=DBL_MAX;
	IMM_label10:for (int x=0;x<INPUT_DEPTH;x++) {
		if (input_max < input[x]) input_max=input[x];
		if (input_min > input[x]) input_min=input[x];
		m0 +=input[x];
	}
	m0=(data_type) m0/INPUT_DEPTH;

	//v0
	v0= pow( (data_type)( input_max - input_min ) / 3 , 2);
	
	//bs

	for (int k=0;k<MAX_COMPONENTS;k++) bS[k]=1;

	//Lambda
	sum_LambdaS=0;
	for (int k=0; k<MAX_COMPONENTS; k++){
		
		LambdaS[k]= Pi[k]*100; //dirichlet
		sum_LambdaS+=LambdaS[k];
	}
	
 
//show(vS,"vS");
//show(bS,"bS");
//show(mS,"mS");
//show(LambdaS,"LambdaS");

 }
 
 data_type IMM::Adaptation(data_type x_new, data_type x_remove, data_type (&mixtures)[3][LIST_DEPTH],int &number_of_components, data_type (&sample_param)[4]){

	//x_new=1.711601224585330883e+01; //just for testing  x_new=2.211601224585330883e+01;
	//data_type samples_[INPUT_DEPTH] = {14.151220496789,47.454596310295,53.961569152495,49.192148348788,16.475781691182,16.269967285869,14.737190467031,50.898556286678,16.079905635544,15.137034179200,14.950776748154,14.555217336074,47.586226431276,15.124072299443,51.204690856086,15.173560277214,18.729110023423,53.854118147522,52.164519573341,14.660659766189,16.342740750267,48.258377449684,14.974085893929,19.539873183333,17.112079917329,53.750329857351,18.258844403380,52.078925788275,55.242069743264,49.329963142527,16.911747848911,48.586085630130,46.166582295940,48.461461782806,14.625934519030,48.031702294963,17.864795292577,15.184636966442,51.860620859623,49.506505029874,15.575046452378,49.954183109161,18.482401919649,16.800377221905,53.289159878769,17.533438128121,51.093074164144,14.798005314404,15.727894900857,47.896205831717,46.836375193986,14.974658499294,16.754355851261,16.813126592968,18.811973656365,16.125639009249,16.319821612765,16.133130475571,15.645815325361,47.959257430490,14.794651735559,15.039496089307,47.715277338648,16.292914471412,52.471429799794,14.587400880866,51.196251473859,15.808955110960,50.165330269878,14.654312574415,51.829477886651,48.309936681925,51.855229997897,51.195168288521,13.848392654607,16.077345876106,15.782352145681,48.163254516647,51.232584188683,19.420798462542,50.587200856152,50.211489693907,14.341445477764,16.142324872446,49.118166116332,18.529415726448,50.551490359777,15.981646587388,47.796856091868,15.032246637422,14.250804226909,50.545898935466,51.342839410461,47.700052578218,15.589066388948,48.015648330821,16.425950238432,50.488325849342,51.891123127578,50.772895443838};

	 data_type N=INPUT_DEPTH; ///<--------------------TODO
	data_type epsilon=1.;

	////////////////////////////////////////////////// Normal Probability distribution update //////////////////
	data_type sample_mean= sample_param[0];
	data_type sample_sigma= sample_param[1];
	data_type sample_squared_sum = sample_param[2];

	data_type x_new_prob_unif= uniform_pdf(x_new,sample_mean,sample_sigma );
	//data_type x_new_prob_norm= norm_pdf(x_new,sample_mean,sample_sigma );

	//update values
	sample_mean = sample_mean + ( x_new - x_remove)/INPUT_DEPTH;
	sample_squared_sum= sample_squared_sum -pow(x_remove,2 ) + pow(x_new,2 );
	sample_sigma = sqrt( sample_squared_sum/INPUT_DEPTH - pow(sample_mean,2 ) ); //according to expected values

	//make the update
	sample_param[0]= sample_mean;
	sample_param[1]= sample_sigma;
	sample_param[2]= sample_squared_sum; 

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	/*
	data_type samples[INPUT_DEPTH];
	samples[INPUT_DEPTH-1]=x_new;

	data_type max_samples= DBL_MIN;
	data_type min_samples= DBL_MAX;

	for (int x=0; x<INPUT_DEPTH; x++) {

		if(x<INPUT_DEPTH-1) samples[x]=input[x+1];

		if (samples[x]<min_samples) min_samples=samples[x];
		if (samples[x]>max_samples) max_samples=samples[x];
	}
*/
	//data_type x_new_pis[LIST_DEPTH]; //list simplification to store.. size is estimate, overflow could occur
	int x_new_pis_count=0;
	
	//data_type prob_x_new[LIST_DEPTH]; //list simplification to store.. size is estimate

	//int x_new_pis_pt=0;

	//int max_samples=0; //initialize max probability 
	int max_prob_x_new_pt;
	int max_new_pis=0;
	int max_epsilon=0;
	
/*	
	while( (x_new-epsilon > min_samples) && (x_new+epsilon < max_samples)  ){
		int num_of_samples=0;
		for (int x=0; x<INPUT_DEPTH; x++) if ( (samples[x]>=x_new-epsilon) && (samples[x]<=x_new+epsilon) ) num_of_samples++;
		x_new_pis[x_new_pis_count++]= num_of_samples; //store and increase pointer
			
		if (num_of_samples>max_new_pis){
			max_epsilon= epsilon;  //store max probability so far
			max_prob_x_new_pt=x_new_pis_count-1; //which element in a row
			max_new_pis=num_of_samples;
		}
 
		epsilon  +=epsilon;
	}
	
	data_type x_new_pi= max_new_pis; //count of occurancies
	data_type epsilon_x_new=pow(2.,(int)max_prob_x_new_pt); //check this---->(int)max_prob_x_new_pt
	data_type max_prob_x_new = max_new_pis / (2 * max_epsilon * N);
*/

	//printf("x_new_pi=%3.4f,epsilon=%3.4f, epsilon_x_new=%3.4f ,max_prob_x_new=%3.4f ,max_prob_x_new_pt=%d \n",x_new_pi,epsilon, epsilon_x_new,max_prob_x_new, max_prob_x_new_pt);

	//data_type D[MAX_COMPONENTS];
	int arg_min_d=0; // Attention points in the number of mixtures D is totaly replaced.
	data_type min_d=DBL_MAX;
	data_type closest_mean,closest_sigma;

	for (int i=0;i<number_of_components;i++) {
		data_type old_mean=mixtures[1][i];
		data_type old_sigma=mixtures[2][i];
		//printf("pow(old_sigma,2)=%6.2f\n",pow(old_sigma,2));
		data_type d_= pow(x_new-old_mean,2)/pow(old_sigma,2);
		data_type d=sqrt(d_);
		if (d<min_d) {min_d=d; arg_min_d=i; closest_mean=old_mean; closest_sigma=old_sigma; }
	}

	int closest=arg_min_d;
	data_type mix_prob_x_new = norm_pdf(x_new,closest_mean,closest_sigma);

 //printf ("x_new_Norm=%8.6f, x_new_Unif=%8.6f, sample_mean=%4.2f, sample_std=%4.2f\n\n",x_new_prob_norm, x_new_prob_unif, sample_mean,sample_std);

 data_type sum_pis=0;
 
 data_type max_prob_x_new= x_new_prob_unif; //******************Applying new scheme
#ifndef SILENT
  printf("\n ---> x_new= %3.3f, closest_mu= %3.3f, closest_sigma= %3.3f\n mix_prob=%3.5f, max_prob=%3.5f\n\n",x_new, closest_mean,closest_sigma,mix_prob_x_new,max_prob_x_new);
#endif
	if ( max_prob_x_new > mix_prob_x_new || mix_prob_x_new<= 0.00000000000001){  //******************Applying new scheme
#ifndef SILENT		
		printf("****** Creating new component ******\n"); 
#endif
		    //data_type eps_pow = int( pow (2*epsilon_x_new,2)-1) /12;
			//printf("epsilon_x_new=%4.2f eps_pow=%3.3f\n", epsilon_x_new,eps_pow); 
		//data_type new_sigma = sqrt( eps_pow );

		data_type new_sigma =3;//abs( abs(x_new - closest_mean)  - 2*closest_sigma )/2;  //******************Applying new scheme
		 
		mixtures[0][number_of_components]= ALPHA; //pis
		mixtures[1][number_of_components]= x_new ; //mean
		mixtures[2][number_of_components]= new_sigma; //sigma

		//printf("new_pi= %3.8f,new_mu= %3.8f, new_sigma= %3.8f  \n\n", mixtures[0][number_of_components],  mixtures[1][number_of_components], mixtures[2][number_of_components]);

		number_of_components++;
	if (number_of_components==LIST_DEPTH) printf("****** MAX components reached LIST FULL ******\n");
	}
	else {//update current mixture
#ifndef SILENT
		printf("***** The new sample is successfully modeled by the mixture *****\n");
#endif
		for (int i=0;i<number_of_components;i++){
				int ownership = 0;
				if (i == closest)  ownership = 1;
			            
			   data_type old_pi	= mixtures[0][i];
			   data_type old_mean = mixtures[1][i];
			   data_type old_sigma = mixtures[2][i];
		   
			   mixtures[0][i] = old_pi + ALPHA * (ownership - old_pi);
			   mixtures[1][i] = old_mean + ownership * ( (x_new - old_mean) / ((old_pi/ALPHA) + 1) );
			   mixtures[2][i] = sqrt(    pow(old_sigma,2) +  ownership*(old_pi/ALPHA * pow (x_new - old_mean,2) /  pow(old_pi/ALPHA + 1,2) - (1. / (old_pi/ALPHA + 1) ) * pow(old_sigma,2)    ) ); 

			   ///printf("mix_pi= %3.6f,mix_mu= %3.6f, mix_sigma= %3.6f  \n", mixtures[0][i],  mixtures[1][i], mixtures[2][i]);
		}
	}

	//normalize pis
	for (int i=0;i<number_of_components;i++)	sum_pis +=  mixtures[0][i];
	for (int i=0;i<number_of_components;i++)	mixtures[0][i] =  mixtures[0][i]/sum_pis ;

	//for (int i=0;i<number_of_components;i++) printf("mix_pi= %3.6f,mix_mu= %3.6f, mix_sigma= %3.6f  \n", mixtures[0][i],  mixtures[1][i], mixtures[2][i]);

	////////////////RETURN STATUS
	//for (int i=0;i<number_of_components;i++) for (int ki=0;ki<3;ki++) mixtures_pass[ki][i]=mixtures[ki][i];
	//no_component_pass=number_of_components;
	////////////////END RETURN STATUS

return sum_pis;

 }
 data_type IMM::uniform_pdf(data_type x, data_type mu, data_type sigma){
	 data_type sig_sqrt_3=sigma*1.7320508075688772935274463415059;
	
		 if ( (x-mu)>= -sig_sqrt_3 && (x-mu) <= sig_sqrt_3){
					return 1./(2*sig_sqrt_3);
		 }
		 else return 0;

 }
 data_type IMM::norm_pdf(data_type x, data_type mu, data_type sigma){

	 //data_type u= (x-mu)/abs(sigma);
	 //data_type y = (1/(sqrt(2*M_PI)*abs(sigma)))*exp(-u*u/2);
	 
	 data_type M_PIx2=(data_type)6.28318530717958647692;
	 data_type sqrt_2_M_PI=(data_type)2.5066282746310005024147107274575;
	 data_type u= (x-mu)/(data_type)sigma;
//printf("sigma %4.4f\n",abs(sigma));
//	 data_type y = 1./ ((data_type)2.5066282746310005024147107274575*abs(sigma) );

	 data_type y =(1./( (data_type)sqrt_2_M_PI*sigma))*exp(-u*u/2);
	 return y;
 }

 void IMM::Estep(void){
	  //printf("Estep started\n");
	//log_pi_s_hat
	for (int k=0;k<MAX_COMPONENTS;k++) log_pi_s_hat[k] = digamma(LambdaS[k])-digamma(sum_LambdaS);
	for (int k=0;k<MAX_COMPONENTS;k++) log_beta_s_hat[k]= digamma(cS[k])-(data_type)log(bS[k]);
	for (int k=0;k<MAX_COMPONENTS;k++) beta_s_mean[k]= cS[k]*bS[k];

//show(log_pi_s_hat,"log_pi_s_hat");
//show(log_beta_s_hat,"log_beta_s_hat");
//show(beta_s_mean,"beta_s_mean");
//show(mS,"mS");
	//temp
	data_type temp[INPUT_DEPTH][MAX_COMPONENTS];
	data_type mS_squared[INPUT_DEPTH];
	data_type vS_squared[INPUT_DEPTH];
	data_type beta_s_mean_0_5[INPUT_DEPTH];

	for (int k=0;k<MAX_COMPONENTS;k++){
		mS_squared[k]=mS[k]*mS[k];
		vS_squared[k]=vS[k]*vS[k];
		beta_s_mean_0_5[k]=(data_type)(-1./2)*beta_s_mean[k];
	}
 
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++){	
			temp[x][k]=(data_type)  beta_s_mean_0_5[k]*( input_squared[x] +  mS_squared[k]  + vS_squared[k] -2*mS[k]*input[x]);
			//printf("x= %d, k=%d, input=%6.3f, input^2= %6.3f, ms^2= %6.3f, vs^2= %6.3f\n ",x,k,input[x], pow(input[x],2), pow(mS[x],2), pow(vS[k],2));
			//getchar();
	}

//show2( temp,"temp");

	//gamma_n_s_hat, epsilon corrected
	data_type exp_temp[INPUT_DEPTH][MAX_COMPONENTS];
	data_type exp_log_pi_s_hat[MAX_COMPONENTS];
	data_type exp_log_beta_s_hat[MAX_COMPONENTS];
	data_type sqrt_exp_log_beta_s_hat[MAX_COMPONENTS];
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++)  exp_temp[x][k]= exp(temp[x][k]);
	for (int k=0;k<MAX_COMPONENTS;k++) exp_log_pi_s_hat[k]= exp(log_pi_s_hat[k]);
	for (int k=0;k<MAX_COMPONENTS;k++) exp_log_beta_s_hat[k]= exp(log_beta_s_hat[k]);
	for (int k=0;k<MAX_COMPONENTS;k++) sqrt_exp_log_beta_s_hat[k]= sqrt(exp_log_beta_s_hat[k]); //big latency


	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++) gamma_n_s_hat[x][k]= exp_temp[x][k]*exp_log_pi_s_hat[k]*sqrt_exp_log_beta_s_hat[k] ;
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++) if (gamma_n_s_hat[x][k]<=0)  gamma_n_s_hat[x][k] = (data_type)FLT_EPSILON;

//show2(gamma_n_s_hat,"gamma_n_s_hat");

	//gamma_n_s_hat again 
	data_type gamma_n_s_hat_SUM_ROW[INPUT_DEPTH]; 
	for (int x=0;x<INPUT_DEPTH;x++) gamma_n_s_hat_SUM_ROW[x]=0;
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++) {
		gamma_n_s_hat_SUM_ROW[x] +=gamma_n_s_hat[x][k];
		//if (gamma_n_s_hat_SUM_ROW[x]==0) gamma_n_s_hat_SUM_ROW[x]=FLT_EPSILON;  //my addition

	}
	
	//gamma_n_s, gamma_n_s epsilon corrected
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++)   {
		gamma_n_s[x][k]=gamma_n_s_hat[x][k] / gamma_n_s_hat_SUM_ROW[x]; //big latency
		if (gamma_n_s[x][k]<=0)  gamma_n_s[x][k] = (data_type)FLT_EPSILON;
	}

//show2(gamma_n_s,"gamma_n_s");
 }

 void IMM::Mstep(void){

	 //pi_s_mean preprocessing
	data_type gamma_n_s_SUM_COLUMN[MAX_COMPONENTS]; 
	for (int k=0;k<MAX_COMPONENTS;k++) gamma_n_s_SUM_COLUMN[k]=0;
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++) gamma_n_s_SUM_COLUMN[k]+= gamma_n_s[x][k];
 
	//pi_s_mean
	for (int k=0;k<MAX_COMPONENTS;k++) {
		pi_s_mean[k]= gamma_n_s_SUM_COLUMN[k]/(data_type)INPUT_DEPTH;
		//if (pi_s_mean[k]==FLT_EPSILON) pi_s_mean[k]=FLT_EPSILON;
	}
//show(pi_s_mean,"pi_s_mean");

	//N_s_mean
	//for (int k=0;k<MAX_COMPONENTS;k++) 
//show(N_s_mean,"N_s_mean");

	//y_s_mean
	for (int k=0;k<MAX_COMPONENTS;k++) {

		N_s_mean[k] = pi_s_mean[k]*(data_type)INPUT_DEPTH;
		LambdaS[k]=LambdaS_[k] = N_s_mean[k] + (data_type)LAMBDA0; //return to gloal
		y_s_mean[k]=y_s_square_mean[k]=0; //init counters
		
	}

	for (int k=0;k<MAX_COMPONENTS;k++) for (int x=0;x<INPUT_DEPTH;x++) y_s_mean[k] +=(gamma_n_s[x][k]*input[x])/INPUT_DEPTH;
//show(y_s_mean,"y_s_mean");

	//y_s_square_mean

	for (int k=0;k<MAX_COMPONENTS;k++) {
		for (int x=0;x<INPUT_DEPTH;x++) y_s_square_mean[k] +=(gamma_n_s[x][k]*input[x]*input[x])/INPUT_DEPTH;

		sigma_s_square_mean[k]= y_s_square_mean[k] + pi_s_mean[k] *(mS[k]*mS[k] + vS[k]) - 2*mS[k]*y_s_mean[k];	
		if (sigma_s_square_mean[k]<=FLT_EPSILON) sigma_s_square_mean[k]=FLT_EPSILON;  //my addition

	}
//show(y_s_square_mean,"y_s_square_mean");

	//sigma_s_square_mean
	//for (int k=0;k<MAX_COMPONENTS;k++) 
//show(sigma_s_square_mean,"sigma_s_square_mean");

	//LambdaS_
	//for (int k=0;k<MAX_COMPONENTS;k++) 
//show(LambdaS_,"LambdaS_");

	//one_div_bS
	for (int k=0;k<MAX_COMPONENTS;k++) {
		one_div_bS[k] = ( ( (data_type)INPUT_DEPTH *sigma_s_square_mean[k])  / (data_type)2 + (data_type) 1./(data_type)BETA   );
		if (one_div_bS[k]<=FLT_EPSILON) one_div_bS[k]=FLT_EPSILON;  //my addition
	}
//show(one_div_bS,"one_div_bS"); 

	//bS_
	for (int k=0;k<MAX_COMPONENTS;k++) bS[k]= (data_type)1. / one_div_bS[k];	//return to global bS
//show(bS,"bS");

	for (int k=0;k<MAX_COMPONENTS;k++) cS[k]= (data_type)(N_s_mean[k] / 2) + (data_type)CE; //return to global cS
//show(cS,"cS");

	for (int k=0;k<MAX_COMPONENTS;k++) {
		m_data_s[k] = y_s_mean[k] / pi_s_mean[k];

	}
//show(m_data_s,"m_data_s");

	for (int k=0;k<MAX_COMPONENTS;k++) tau_data_s[k] = N_s_mean[k] * bS[k] * cS[k];
//show(tau_data_s,"tau_data_s");

	for (int k=0;k<MAX_COMPONENTS;k++) {
		tau_s[k] = (1. / v0) + tau_data_s[k];
		//if (tau_s[k]==0) tau_s[k]=FLT_EPSILON;  //my addition
	}
//show(tau_s,"tau_s");

	for (int k=0;k<MAX_COMPONENTS;k++) mS[k]= m0/(v0*tau_s[k]) + (tau_data_s[k] * m_data_s[k]) / tau_s[k]; //return to global
//show(mS,"mS");

	//DONT" uncomment this.. it was false updatedfor (int k=0;k<MAX_COMPONENTS;k++) vS[k]= (1./bS[k]) / (INPUT_DEPTH*pi_s_mean[k]); //return to global
//show(vS,"vS");

 }

 void IMM::Fit(data_type (&mixtures)[3][LIST_DEPTH],int &number_of_components){
	 //printf("Fit started\n");
	 bool converged =false;
	 int counter = 0;

	 data_type  Lambda_s_old[MAX_COMPONENTS], bS_old[MAX_COMPONENTS], cS_old[MAX_COMPONENTS], mS_old[MAX_COMPONENTS], vS_old[MAX_COMPONENTS];
 
	 while(!converged){
		
//printf("\n---------Iteration= %d\n", counter);

		// show2(gamma_n_s,"gamma_n_s");

			for (int k=0;k<MAX_COMPONENTS;k++) {
				Lambda_s_old[k]=LambdaS[k];
				bS_old[k]=bS[k];
				cS_old[k]=cS[k];
				mS_old[k]=mS[k];
				vS_old[k]=vS[k];
		
			}
			 Estep();  //updates gamma_n_s
				//show(bS_old,"bS_old");
				//show(cS_old,"cS_old");
				//show(mS_old,"mS_old");
				//show(vS_old,"vS_old");

//show(mS_old,"mS_old");
				
		Mstep();  //return Lambda_s , bS , cS , mS , vS
//show(bS,"bS_new");
//show(cS,"cS_new");
//show(mS,"mS_new");
//show(vS,"vS_new");

 int hits=0;
		bool temp_converged;
		for (int k=0;k<MAX_COMPONENTS;k++) {
			if(abs(mS[k]- mS_old[k]) < (data_type)CRIT) {
			
			hits++;
			//printf( "   criterion[%d]=%6.6f\n",k,fabs(mS[k]- mS_old[k] )  );

			}
			//temp_converged=true;

			if ( hits) temp_converged=true ;
			else temp_converged=false;
		}
 
		if(temp_converged) converged=true;
		counter++;
#ifdef TESTING
			if (counter>100){ printf("Taking too long to converge...breaking\n"); break;}

			//getchar();
			//printf("Press any key");
			
#endif
		} //end while


	//sigma_est
	for (int k=0;k<MAX_COMPONENTS;k++) sigma_est[k]= sqrt(vS[k]);

	//check some inconsistencies found, but only in non-relevant components
//
	//show(sigma_est,"sigma_est"); 

	//components
	 
	for (int k=0;k<MAX_COMPONENTS;k++) sigma_est[k]= sqrt(vS[k]);

	//pi_est preprocessing
	data_type gamma_n_s_SUM_COLUMN[MAX_COMPONENTS]; //temp variable
	for (int k=0;k<MAX_COMPONENTS;k++) gamma_n_s_SUM_COLUMN[k]=0;
	for (int x=0;x<INPUT_DEPTH;x++) for (int k=0;k<MAX_COMPONENTS;k++) gamma_n_s_SUM_COLUMN[k]+= gamma_n_s[x][k];

	//pi_est
	number_of_components=0;
	data_type normalized_factor_sum=0; //init outsided the loop

	for (int k=0;k<MAX_COMPONENTS;k++)pi_est[k]= gamma_n_s_SUM_COLUMN[k]/INPUT_DEPTH;
//show(pi_est,"pi_est");

	for (int k=0;k<MAX_COMPONENTS;k++) {

		if (pi_est[k] > (data_type) 5e-2) {
			components[k]=true; 
			number_of_components ++;
			normalized_factor_sum +=pi_est[k];  // only if component is true
		}
		else components[k]=false;
	}

//show(pi_est,"pi_est");

	// normalize_factor
	data_type normalize_factor= (data_type)1./ normalized_factor_sum;

	//pis
	for (int k=0;k<MAX_COMPONENTS;k++) pis[k]=pi_est[k] * normalize_factor;
//show(pis,"pis");
	int cc=0;
	//prepare mixtures matrix for adaptation
	for (int k=0;k<MAX_COMPONENTS;k++) if (components[k]){
		mixtures[0][cc]= pis[k];// pis value 
		mixtures[1][cc]= mS[k];// mean value
		mixtures[2][cc]= sigma_est[k];// sigma value
		cc++;
	}
	//output messages
#ifndef SILENT
	printf("The algorithm converged after: %d iterations\n", counter);
	printf("Number of components found =%d \n\n", number_of_components);
	
	for (int k=0;k<MAX_COMPONENTS;k++) if (components[k]) printf("Component %d: pi: %3.6f , mean: %3.6f , std: %3.6f \n", k, pi_est[k], mS[k], sigma_est[k]);
	printf("\n");

#endif
 }

 void IMM:: kmeans(data_type data[INPUT_DEPTH] ){

//printf("Kmeans started\n");
//for (int i=0; i<INPUT_DEPTH; i++) printf("data[%d]=%3.8f\n", i, data[i]);

data_type old_error=DBL_MAX;
data_type error=DBL_MAX;

data_type centroids_temp[MAX_COMPONENTS][KMEANS_DIM]; //
/*
  data_type centroids_init[MAX_COMPONENTS]={
1.812820512820511709e+01,
6.544444444444444287e+01,
3.175000000000000000e+01,
8.000000000000000000e+01,
5.285714285714286120e+01
};
*/
//initialize centroids
int rand_no;
int step=0;

float randA[20]={177, 4, 187, 52, 106, 131, 239, 154, 135, 109, 91, 209, 83, 230, 128, 214, 210, 248, 233, 173};
 
   for (int i = 0; i < MAX_COMPONENTS; i++) {
	//centroids[i][0]=0;
	
	///if (step<INPUT_DEPTH) centroids[i][0]=data[step]; else printf("\n error in step");
	//centroids[0][0]=36;
	 centroids[i][0]=  randA[i] ; // rand() % 10 - rand() % 10 );

	 step += INPUT_DEPTH/MAX_COMPONENTS;
	 //printf("RAND1=%3.4f\n",  rand() % 10 - rand() % 10  );
	 //rand_no= rand() % 10 - rand() % 10 ;

	 //printf("rand  %4.2f\n", rand_no);

	 
	 //printf("INIT: centroids[%d]=%2.3f \n", i,centroids[i][0]);
	 
	 //getchar();
	 //step= rand() % 100; //step + INPUT_DEPTH/MAX_COMPONENTS; ///  //equally assign centroids
		//centroids[i][0]= data[rand() % 99];
   }
     
 int iterations=0;
  do {
	iterations++;
	old_error = error;
	error = 0;

    for (int k = 0; k < MAX_COMPONENTS; k++) {
		clusters_size[k] = 0;
		centroids_temp[k][0] = inertia[k][0]=0; //resets temp centroids
    }

//*********************
      for (int x= 0; x < INPUT_DEPTH; x++) {

         // find the closest cluster
         data_type min_distance = DBL_MAX; //init at max distance
         for (int  k = 0; k < MAX_COMPONENTS; k++) {
            data_type distance = 0;
           
			//distance += pow(x- centroids[k][0], 2);	//adding euclidian distances for x dimention
			distance += (data_type) (data[x] - centroids[k][0])*(data[x] - centroids[k][0]); //addiing euclidian distances for y dimention
           
			if (distance < min_distance  ) {
				labels[x] = k; //store for each data point (store x ) each label
				min_distance = distance;
			} //end if

         } //end for k

			 // update size and temp centroid of the destination cluster 
			 centroids_temp[labels[x]] [0] += data[x];
			 //if(labels[x]==0) printf ("centroids_temp[%d]=%6.2f\n", labels[x],centroids_temp[labels[x]][0] );
			 //if ( labels[x]>= MAX_COMPONENTS) printf ("labelx=%d\n", labels[x] );
			 clusters_size[labels[x]]++;

			 //inertia[labels[x]][0]+=min_distance;
				//if(labels[x]==0) printf ("x=%d, data[x]=%4.2f ,inertia[%d]=%6.2f\n", x, data[x], labels[x],inertia[labels[x]][0] );

         /* update standard error */
			error += min_distance;
      } //end for x


	  /* update all centroids */

      for (int i = 0; i < MAX_COMPONENTS; i++) { 

         if(clusters_size[i]) {		 
			 centroids[i][0]=centroids_temp[i][0] / clusters_size[i];
			  //printf("DEBUG: centroids_temp[%d]=%2.1f , clusters_size[%d]=%2.1f \n", i,centroids_temp[i][0], i,clusters_size[i]);
		 }

		 else  {
			 centroids[i][0]= 0;//centroids_temp[i][0];	
			  // printf("DEBUG: centroids_temp[%d]=%2.1f , clusters_size[%d]=%2.1f \n", i,centroids_temp[i][0],i, clusters_size[i]);
		 }

		// centroids[i][0]=centroids_init[i]; // <*********************DEBUG TESTING*********************
         
      } //end for


   } while (abs(error - old_error) > (data_type)KMEANS_MAX_ERR);//end while

 //********************** Variance ********************
	   for (int k = 0; k < MAX_COMPONENTS; k++){
		 data_type sum_c=0;
		 for (int x=0;x<INPUT_DEPTH; x++){
		   if(labels[x]==k){
			   sum_c = sum_c+pow(data[x]- centroids[k][0],2);

		   }
		   inertia[k][0]=sum_c;
		}
	   }
//************************************************
  //
 // printf ("\nKMeans Iterations=%d\n\n", iterations);
 //
//for (int i=0; i<MAX_COMPONENTS; i++) printf("custer_size= %d, cendroids[%d]=%8.6f, inertia[%d]=%4.6f  \n", clusters_size[i], i, centroids[i][0], i, inertia[i][0]);
//getchar();
}

 void IMM::show(data_type x[MAX_COMPONENTS], char str[12]){

	 for (int k=0; k<MAX_COMPONENTS; k++) {printf ("%s[%2d]= %4.18f \n", str,k,x[k]);  }
	 printf("\n Press any key and enter");
#ifndef __SYNTHESIS__
	 getchar();
#endif	 
}

 void IMM::show2(data_type x[INPUT_DEPTH][MAX_COMPONENTS], char str[12])
 {
	 int count=0;
	 //for (int k=0; k<MAX_COMPONENTS; k++) printf ("var[%d]= %3.4f \n", k,x[k]);
	// for (int k=0; k<MAX_COMPONENTS; k++) printf ("%s[%2d]", str,k); 

	// printf ("%s",str);
	  for(int k=0; k<MAX_COMPONENTS; k++) for(int xi=0;xi<INPUT_DEPTH;xi++) {
		printf ("%s[%2d,%2d]= %6.12f \n", str,xi,k,x[xi][k]);
		//count++;
#ifndef __SYNTHESIS__
		if (xi == INPUT_DEPTH-2) { getchar(); printf("\n Press any key and enter\n");}
#endif	 
	 }
	 
}
/* digamma.c
 *
 * Mark Johnson, 2nd September 2007
 *
 * Computes the Ø(x) or digamma function, i.e., the derivative of the 
 * log gamma function, using a series expansion.
 *
 * Warning:  I'm not a numerical analyst, so I may have made errors here!
 *
 * The parameters of the series were computed using the Maple symbolic
 * algebra program as follows:
 *
 * series(Psi(x+1/2), x=infinity, 21);
 *
 * which produces:
 *
 *  ln(x)+1/(24*x^2)-7/960/x^4+31/8064/x^6-127/30720/x^8+511/67584/x^10-1414477/67092480/x^12+8191/98304/x^14-118518239/267386880/x^16+5749691557/1882718208/x^18-91546277357/3460300800/x^20+O(1/(x^21)) 
 *
 * It looks as if the terms in this expansion *diverge* as the powers
 * get larger.  However, for large x, the x^-n term will dominate.
 *
 * I used Maple to examine the difference between this series and
 * Digamma(x+1/2) over the range 7 < x < 20, and discovered that the
 * difference is less that 1e-8 if the terms up to x^-8 are included.
 * This determined the power used in the code here.  Of course,
 * Maple uses some kind of approximation to calculate Digamma,
 * so all I've really done here is find the smallest power that produces
 * the Maple approximation; still, that should be good enough for our
 * purposes.
 *
 * This expansion is accurate for x > 7; we use the recurrence 
 *
 * digamma(x) = digamma(x+1) - 1/x
 *
 * to make x larger than 7.
 */

//#include <assert.h>
//#include <math.h>

data_type IMM::digamma(data_type x) {
	data_type tx=x;
  data_type result = 0, xx, xx2, xx4;

  //assert(x > 0);
  for ( ; x < 7; ++x)
    result -= (data_type)1/x;
  x -= (data_type)1.0/(data_type)2.0;
  xx = (data_type)1.0/x;
  xx2 = (data_type)xx*xx;
  xx4 = (data_type)xx2*xx2;
  result += (data_type)log((data_type)x)+(data_type)(data_type)(1./24.)*xx2-(data_type)(7.0/960.0)*xx4+(data_type)(31.0/8064.0)*xx4*xx2-(data_type)(127.0/30720.0)*xx4*xx4;
    //if (tx>20) printf ("------------------assert error in digamma x=%6.2f, psi=%3.16f\n",tx,result);
  return result;
}



