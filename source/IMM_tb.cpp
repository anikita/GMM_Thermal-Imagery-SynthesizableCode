#include "define.h"

#ifdef TESTING
#ifdef OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;
#endif

#endif


data_type IMM_top_level(int THREAD_N);
data_type IMM_top_level_synth(void);

using namespace std;
#ifdef OPENCV
	void mog2();
#endif

int main()
{

 
data_type a;
 

#ifdef TESTING
 
	 a=IMM_top_level(4);
 	 
	 //a=IMM_top_level(1);
	 getchar();

	//******************************opencv testing *****************
/*
   CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    capture = cvCaptureFromCAM( 0 ); //0=default, -1=any camera, 1..99=your camera
    if(!capture) cout << "No camera detected" << endl;

    cvNamedWindow( "result", 1 );

    if( capture )
    {
        cout << "In capture ..." << endl;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            if( frame.empty() )
                break;
            if( iplImg->origin == IPL_ORIGIN_TL )
                frame.copyTo( frameCopy );
            else
                flip( frame, frameCopy, 0 );

            if( waitKey( 40 ) >= 0 )
                cvReleaseCapture( &capture );
        }

        waitKey(0);

    cvDestroyWindow("result");
	}

//*****************************end opencv testing **************

*/



#else
a=IMM_top_level_synth();
#endif
	// getchar();
printf("returning all good\n");
return 0;
}
