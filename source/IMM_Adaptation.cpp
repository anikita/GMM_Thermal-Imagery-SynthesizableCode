#include "IMM_Adaptation.h"


IMM_Adaptation::IMM_Adaptation (int a){
	a=a;
}
 data_type IMM_Adaptation:: Adaptation(data_type x_new, data_type x_remove, data_type (&mixtures)[3][LIST_DEPTH],int &number_of_components, data_type (&sample_param)[4]){


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

		data_type new_sigma =2; //abs( abs(x_new - closest_mean)  - 2*closest_sigma )/4;  //******************Applying new scheme
		 
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

  data_type IMM_Adaptation::uniform_pdf(data_type x, data_type mu, data_type sigma){
	 data_type sig_sqrt_3=sigma*1.7320508075688772935274463415059;
	 data_type sig_sqrt_3x2=sigma*3.4641016151377545870548926830118;
	
		 if ( (x-mu)>= -sig_sqrt_3 && (x-mu) <= sig_sqrt_3){
					return 1./(sig_sqrt_3x2);
		 }
		 else return 0;

 }
 data_type IMM_Adaptation::norm_pdf(data_type x, data_type mu, data_type sigma){
	 data_type M_PIx2=(data_type)6.28318530717958647692;
	 data_type sqrt_2_M_PI=(data_type)2.5066282746310005024147107274575;
	 data_type u= (x-mu)/(data_type)sigma;
//printf("sigma %4.4f\n",abs(sigma));
//	 data_type y = 1./ ((data_type)2.5066282746310005024147107274575*abs(sigma) );

	 data_type y =(1./( (data_type)sqrt_2_M_PI*sigma))*exp(-u*u/2);
	 return y;
 }
