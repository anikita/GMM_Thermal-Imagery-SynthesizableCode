#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>

#include <iostream>
#include <vector>
#include <queue> 

using namespace std;
using namespace cv;

#include "define.h"
#include "dirent.h"
#include "IMM.h"
#include "kmeans_pp.h"

struct pixel_model{
	data_type mixtures[3][MAX_COMPONENTS];
	int components;
	data_type sample_params[4];
	data_type history[INPUT_DEPTH];
};

int testk();
void mog2()
{
 

	getchar;
    cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
  
	Mat * image_array= new Mat[105];
	pixel_model * BG_P =new pixel_model[400*400];
	std::queue <data_type> * Frame_queue= new queue <data_type> [400*400];
	///*********************Open Directory *******************

	char dirn[50]="osu";
		
        DIR *dir = NULL;
        struct dirent *drnt = NULL;

        printf("Input dir name: ");
        //gets(dirn);


        dir=opendir(dirn);
		int count=0;
		if(dir)
        {
			
			 printf("output:\n");
			 while(drnt = readdir(dir))
                {
					char path[50];
					strncpy( path, dirn,sizeof(path) ) ;
					strcat( path, "\\" ) ;
					strcat( path, drnt->d_name);

					//printf("%s\n", path);
					Mat temp_image= imread(path, CV_LOAD_IMAGE_GRAYSCALE);
					if(!temp_image.data)   cout <<  "Could not open or find the image" << std::endl;

					else {
							image_array[count] =  imread(path, CV_LOAD_IMAGE_GRAYSCALE);
							 //cout<<"original image channels: "<<image_array[count].channels()<<"gray image channels: "<<image_array[count].channels()<<endl; 
							printf("%s\n", path);
							count++;

						}
						
					if (count>INPUT_DEPTH) { printf("100 images history reached, see you in a next loop..\n"); break;}
			 }
			
		}

	printf("Loaded %d images..\n",count);
//*****************************end open directory **************************************

//** bulid history buffers

FILE*	wfile = fopen("pix_data.txt", "w"); if (wfile == NULL)  printf("Error Writing File: gfile1\n");

for(int i=0; i<image_array[0].rows; i++){
   for(int j=0; j<image_array[0].cols; j++){

	   for (int k=0; k<count; k++) { 

		   data_type temp_data = (data_type) image_array[k].at<unsigned char>(i,j);
		   int pixel_pos=i*image_array[0].cols + j;

		   BG_P[pixel_pos].history[k] = temp_data;  //history passes as an array-- it is synthesizable type
		  // Frame_queue[pixel_pos].push(temp_data);

		 //Scalar temp= image_array[k].at<unsigned char>(i,j);
		// printf( "Raw Pixel=%d,  Image:%d  Data:%d \n" ,i*image_array[0].cols + j, k,   image_array[k].at<unsigned char>(i,j)  );

		//namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
		//imshow( "Display window", image_array[k] );       // Show our image inside it.
		//waitKey(0); //wait infinite time for a keypress
	   }

	 //  getchar();

   }
}
    //namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    //imshow( "Display window", image_array[0] );       // Show our image inside it.


for (int pixel=18411; pixel < 18455+10 ; pixel++){
	printf("pixel: %d ... \n",pixel);
	//for (int ki=0; ki<100;ki++) printf("intensity[%d]:%7.3f  \n",ki, BG_P[pixel].history[ki]);
	for (int ki=0; ki<100;ki++) fprintf(wfile,"%3.1f , ",BG_P[pixel].history[ki]);


IMM  imm0(BG_P[pixel].history, BG_P[pixel].sample_params);
imm0.Fit(BG_P[pixel].mixtures,BG_P[pixel].components);


//for (int k=0;k<BG_P[pixel].components;k++)   printf("OUTside1 Component %d: pi: %3.6f , mean: %3.6f , std: %3.6f \n", k, BG_P[pixel].mixtures[0][k], BG_P[pixel].mixtures[1][k], BG_P[pixel].mixtures[2][k]);

//Frame_queue[pixel].push(2.2);
data_type a;
 // a= imm0.Adaptation(2.2, Frame_queue[pixel].front(), BG_P[pixel].mixtures, BG_P[pixel].components, BG_P[pixel].sample_params);
//Frame_queue[pixel].pop();

//for (int k=0;k<BG_P[pixel].components;k++)   printf("OUTside2 Component %d: pi: %3.6f , mean: %3.6f , std: %3.6f \n", k, BG_P[pixel].mixtures[0][k], BG_P[pixel].mixtures[1][k], BG_P[pixel].mixtures[2][k]);
//getchar();

}
	fclose (wfile); 
   // waitKey(0); //wait infinite time for a keypress
	delete[] image_array;
	delete[] BG_P;
	//delete[] Frame_queue;

     
}