
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "features2d/features2d.hpp"
#include <iostream>  
      
using namespace std;  
using namespace cv;  
      
const string winName = "image";  

    int main( int argc, char** argv )  
    {    
        string filename = argv[1];  
        Mat image = imread( filename );  
		Point p1 = Point(1, 1); 
		Point p2 = Point(image.cols-1, image.rows-1); 
		Rect rect(p1,p2);
		Mat mask; Mat bgdModel, fgdModel; 
		grabCut( image, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_RECT );
        grabCut( image, mask, rect, bgdModel, fgdModel, 1 );  //执行grabCut算法 
		grabCut( image, mask, rect, bgdModel, fgdModel, 1 );
		grabCut( image, mask, rect, bgdModel, fgdModel, 1 );
		Mat binMask;  Mat res2; 
		binMask = mask & 1;  
		image.copyTo( res2, binMask ); 
		imshow( winName, res2 );  
		waitKey();

        return 0;  
    }

























//#include "opencv2/highgui/highgui.hpp"
//
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include "features2d/features2d.hpp"
//#include <iostream>  
//
//      
//using namespace std;  
//using namespace cv;  
//      
//const string winName = "image";  
//
//    int main( int argc, char** argv )  
//    {    
//        string filename = argv[1];  
//		//int a[]={1,2,3,4};
//		//Mat test=Mat(1,4,CV_8UC1,a);//CV_8UC3:(3-1)*2^3=16;CV_8UC2:(2-1)*2^3=8;CV_8UC1:(1-1)*2^3=0
//		//cout<<test.type()<<endl;
//        Mat image = imread( filename );  
//		Mat mask_temp=imread("hello_HC.png");
//
//		//threshold( mask_temp, mask_temp, 200, 255,THRESH_BINARY );
//		
//		imshow("hello",mask_temp);//作为模板
//		cvtColor(mask_temp,mask_temp,CV_BGR2GRAY );
//		
//		//下面一段代码为了找到95%的亮度阈值
//		//vector<int> result(256);
//		//for(int i=0;i<mask_temp.rows;i++)
//		//	for(int j=0;j<mask_temp.cols;j++)
//		//		result[mask_temp.at<uchar>(i,j)]++; 
//		//int sum=0,threshold_temp;
//		//for(int i=255;i>0;i--)
//		//{
//		//	sum+=result[i];
//		//	if(sum>mask_temp.rows*mask_temp.cols*0.05)
//		//		{
//		//			threshold_temp=i;
//		//			break;
//		//		}
//		//}
//		//
//		threshold( mask_temp, mask_temp, 70, 255,THRESH_BINARY );		
//		int erosion_size=10;
//		Mat element = getStructuringElement( MORPH_RECT,Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
//		Mat mask_erode;
//		erode( mask_temp, mask_erode, element );
//		imshow("erode",mask_erode);//作为模板
//		//for(int i=0;i<mask_temp.rows;i++)
//		//	for(int j=0;j<mask_temp.cols;j++)
//		//		if(mask_erode.at<uchar>(i,j)==255)
//		//			mask_temp.at<uchar>(i,j)=GC_FGD;
//		//		else 
//		//			mask_temp.at<uchar>(i,j)=GC_PR_FGD  ;
//
//
//		//Mat element = getStructuringElement( MORPH_RECT,Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
//		Mat mask_dilate;
//		dilate( mask_temp, mask_dilate, element );
//		imshow("dilate",mask_dilate);//作为模板
//
//		for(int i=0;i<mask_temp.rows;i++)
//			for(int j=0;j<mask_temp.cols;j++)
//				if(mask_dilate.at<uchar>(i,j)==0)
//					mask_temp.at<uchar>(i,j)=GC_BGD;
//				else if(mask_erode.at<uchar>(i,j)==255)
//					mask_temp.at<uchar>(i,j)=GC_FGD;
//				else 
//					mask_temp.at<uchar>(i,j)=GC_PR_BGD   ;
//		
//
//		grabCut( image, mask_temp, Rect(), Mat(), Mat(), 0, GC_INIT_WITH_MASK  );
//		//		imshow("hello3",mask_temp);//作为模板
//		////mask_temp&=1;
//		//Point p1 = Point(84, 110); 
//		//Point p2 = Point(245, 285); 
//		//Rect rect(p1,p2);
//		Mat mask; Mat bgdModel, fgdModel; 
//		
//        grabCut( image, mask_temp, Rect(), Mat(), Mat(), 1 );  //执行grabCut算法 
//		  //grabCut( image, mask_temp, Rect(), bgdModel, fgdModel, 1 );  //执行grabCut算法 
//		//threshold( mask_temp, mask_temp, 1, 255,THRESH_BINARY );	
//		//int erosion_size=4;
//		//Mat element = getStructuringElement( MORPH_ELLIPSE,Size( 2*erosion_size + 1, 2*erosion_size+1 ),Point( erosion_size, erosion_size ) );
//		//erode( mask_temp, mask_temp, element );
//		//imshow("hello2",mask_temp);//作为模板
//
//		Mat binMask;  Mat res2; 
//		binMask = mask_temp & 1;  
//		image.copyTo( res2, binMask ); 
//		imshow( winName, res2 );  
//		waitKey();
//
//        return 0;  
//    }   










//#include "precomp.hpp"
//#include "gcgraph.hpp"
//#include <limits> 
//
//      
//using namespace std;  
//
//using namespace cv;
//
///*
//This is implementation of image segmentation algorithm GrabCut described in
//"GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
//Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
// */
//
///*
// GMM - Gaussian Mixture Model
//*/
//class GMM
//{
//public:
//    static const int componentsCount = 5;
//
//    GMM( Mat& _model );
//    double operator()( const Vec3d color ) const;
//    double operator()( int ci, const Vec3d color ) const;
//    int whichComponent( const Vec3d color ) const;
//
//    void initLearning();
//    void addSample( int ci, const Vec3d color );
//    void endLearning();
//
//private:
//    void calcInverseCovAndDeterm( int ci );
//    Mat model;
//    double* coefs;
//    double* mean;
//    double* cov;
//
//    double inverseCovs[componentsCount][3][3];
//    double covDeterms[componentsCount];
//
//    double sums[componentsCount][3];
//    double prods[componentsCount][3][3];
//    int sampleCounts[componentsCount];
//    int totalSampleCount;
//};
//
//GMM::GMM( Mat& _model )
//{
//    const int modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
//    if( _model.empty() )
//    {
//        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
//        _model.setTo(Scalar(0));
//    }
//    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
//        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );
//
//    model = _model;
//
//    coefs = model.ptr<double>(0);
//    mean = coefs + componentsCount;
//    cov = mean + 3*componentsCount;
//
//    for( int ci = 0; ci < componentsCount; ci++ )
//        if( coefs[ci] > 0 )
//             calcInverseCovAndDeterm( ci );
//}
//
//double GMM::operator()( const Vec3d color ) const
//{
//    double res = 0;
//    for( int ci = 0; ci < componentsCount; ci++ )
//        res += coefs[ci] * (*this)(ci, color );
//    return res;
//}
//
//double GMM::operator()( int ci, const Vec3d color ) const
//{
//    double res = 0;
//    if( coefs[ci] > 0 )
//    {
//		CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
//        Vec3d diff = color;
//        double* m = mean + 3*ci;
//        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
//        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
//                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
//                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
//        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
//    }
//    return res;
//}
//
//int GMM::whichComponent( const Vec3d color ) const
//{
//    int k = 0;
//    double max = 0;
//
//    for( int ci = 0; ci < componentsCount; ci++ )
//    {
//		double p = (*this)( ci, color );
//        if( p > max )
//        {
//            k = ci;
//            max = p;
//        }
//    }
//    return k;
//}
//
//void GMM::initLearning()
//{
//    for( int ci = 0; ci < componentsCount; ci++)
//    {
//        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
//        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
//        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
//        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
//        sampleCounts[ci] = 0;
//    }
//    totalSampleCount = 0;
//}
//
//void GMM::addSample( int ci, const Vec3d color )
//{
//    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
//    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
//    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
//    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
//    sampleCounts[ci]++;
//    totalSampleCount++;
//}
//
//void GMM::endLearning()
//{
//    const double variance = 0.01;
//    for( int ci = 0; ci < componentsCount; ci++ )
//    {
//        int n = sampleCounts[ci];
//        if( n == 0 )
//            coefs[ci] = 0;
//        else
//        {
//            coefs[ci] = (double)n/totalSampleCount;
//
//            double* m = mean + 3*ci;
//            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;
//
//            double* c = cov + 9*ci;
//            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
//            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
//            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];
//
//            double dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
//            if( dtrm <= std::numeric_limits<double>::epsilon() )
//            {
//                // Adds the white noise to avoid singular covariance matrix.
//                c[0] += variance;
//                c[4] += variance;
//                c[8] += variance;
//            }
//
//            calcInverseCovAndDeterm(ci);
//        }
//    }
//}
//
//void GMM::calcInverseCovAndDeterm( int ci )
//{
//    if( coefs[ci] > 0 )
//    {
//        double *c = cov + 9*ci;
//        double dtrm =
//              covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
//
//        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );
//        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
//        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
//        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
//        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
//        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
//        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
//        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
//        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
//        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
//    }
//}
//
///*
//  Calculate beta - parameter of GrabCut algorithm.
//  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
//*/
//static double calcBeta( const Mat& img )
//{
//    double beta = 0;
//    for( int y = 0; y < img.rows; y++ )
//    {
//        for( int x = 0; x < img.cols; x++ )
//        {
//            Vec3d color = img.at<Vec3b>(y,x);
//            if( x>0 ) // left
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
//                beta += diff.dot(diff);
//            }
//            if( y>0 && x>0 ) // upleft
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
//                beta += diff.dot(diff);
//            }
//            if( y>0 ) // up
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
//                beta += diff.dot(diff);
//            }
//            if( y>0 && x<img.cols-1) // upright
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
//                beta += diff.dot(diff);
//            }
//        }
//    }
//    if( beta <= std::numeric_limits<double>::epsilon() )
//        beta = 0;
//    else
//        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
//
//    return beta;
//}
//
///*
//  Calculate weights of noterminal vertices of graph.
//  beta and gamma - parameters of GrabCut algorithm.
// */
//static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
//{
//    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
//    leftW.create( img.rows, img.cols, CV_64FC1 );
//    upleftW.create( img.rows, img.cols, CV_64FC1 );
//    upW.create( img.rows, img.cols, CV_64FC1 );
//    uprightW.create( img.rows, img.cols, CV_64FC1 );
//    for( int y = 0; y < img.rows; y++ )
//    {
//        for( int x = 0; x < img.cols; x++ )
//        {
//            Vec3d color = img.at<Vec3b>(y,x);
//            if( x-1>=0 ) // left
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y,x-1);
//                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
//            }
//            else
//                leftW.at<double>(y,x) = 0;
//            if( x-1>=0 && y-1>=0 ) // upleft
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x-1);
//                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
//            }
//            else
//                upleftW.at<double>(y,x) = 0;
//            if( y-1>=0 ) // up
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x);
//                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
//            }
//            else
//                upW.at<double>(y,x) = 0;
//            if( x+1<img.cols && y-1>=0 ) // upright
//            {
//                Vec3d diff = color - (Vec3d)img.at<Vec3b>(y-1,x+1);
//                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
//            }
//            else
//                uprightW.at<double>(y,x) = 0;
//        }
//    }
//}
//
///*
//  Check size, type and element values of mask matrix.
// */
//static void checkMask( const Mat& img, const Mat& mask )
//{
//    if( mask.empty() )
//        CV_Error( CV_StsBadArg, "mask is empty" );
//    if( mask.type() != CV_8UC1 )
//        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
//    if( mask.cols != img.cols || mask.rows != img.rows )
//        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
//    for( int y = 0; y < mask.rows; y++ )
//    {
//        for( int x = 0; x < mask.cols; x++ )
//        {
//            uchar val = mask.at<uchar>(y,x);
//            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
//                CV_Error( CV_StsBadArg, "mask element value must be equel"
//                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
//        }
//    }
//}
//
///*
//  Initialize mask using rectangular.
//*/
//static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
//{
//    mask.create( imgSize, CV_8UC1 );
//    mask.setTo( GC_BGD );
//
//    rect.x = max(0, rect.x);
//    rect.y = max(0, rect.y);
//    rect.width = min(rect.width, imgSize.width-rect.x);
//    rect.height = min(rect.height, imgSize.height-rect.y);
//
//    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
//}
//
///*
//  Initialize GMM background and foreground models using kmeans algorithm.
//*/
//static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
//{
//    const int kMeansItCount = 10;
//    const int kMeansType = KMEANS_PP_CENTERS;
//
//    Mat bgdLabels, fgdLabels;
//    vector<Vec3f> bgdSamples, fgdSamples;
//    Point p;
//    for( p.y = 0; p.y < img.rows; p.y++ )
//    {
//        for( p.x = 0; p.x < img.cols; p.x++ )
//        {
//            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
//                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
//            else // GC_FGD | GC_PR_FGD
//                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
//        }
//    }
//    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
//    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
//    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
//            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
//    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
//    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
//            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
//
//    bgdGMM.initLearning();
//    for( int i = 0; i < (int)bgdSamples.size(); i++ )
//        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
//    bgdGMM.endLearning();
//
//    fgdGMM.initLearning();
//    for( int i = 0; i < (int)fgdSamples.size(); i++ )
//        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
//    fgdGMM.endLearning();
//}
//
///*
//  Assign GMMs components for each pixel.
//*/
//static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
//{
//    Point p;
//    for( p.y = 0; p.y < img.rows; p.y++ )
//    {
//        for( p.x = 0; p.x < img.cols; p.x++ )
//        {
//            Vec3d color = img.at<Vec3b>(p);
//			compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
//                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
//        }
//    }
//}
//
///*
//  Learn GMMs parameters.
//*/
//static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
//{
//    bgdGMM.initLearning();
//    fgdGMM.initLearning();
//    Point p;
//    for( int ci = 0; ci < GMM::componentsCount; ci++ )
//    {
//        for( p.y = 0; p.y < img.rows; p.y++ )
//        {
//            for( p.x = 0; p.x < img.cols; p.x++ )
//            {
//                if( compIdxs.at<int>(p) == ci )
//                {
//                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
//                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
//                    else
//                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
//                }
//            }
//        }
//    }
//    bgdGMM.endLearning();
//    fgdGMM.endLearning();
//}
//
///*
//  Construct GCGraph
//*/
//static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
//                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
//                       GCGraph<double>& graph )
//{
//    int vtxCount = img.cols*img.rows,
//        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
//    graph.create(vtxCount, edgeCount);
//    Point p;
//    for( p.y = 0; p.y < img.rows; p.y++ )
//    {
//        for( p.x = 0; p.x < img.cols; p.x++)
//        {
//            // add node
//            int vtxIdx = graph.addVtx();
//            Vec3b color = img.at<Vec3b>(p);
//
//            // set t-weights
//            double fromSource, toSink;
//            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
//            {
//                fromSource = -log( bgdGMM(color) );
//                toSink = -log( fgdGMM(color) );
//            }
//            else if( mask.at<uchar>(p) == GC_BGD )
//            {
//                fromSource = 0;
//                toSink = lambda;
//            }
//            else // GC_FGD
//            {
//                fromSource = lambda;
//                toSink = 0;
//            }
//            graph.addTermWeights( vtxIdx, fromSource, toSink );
//
//            // set n-weights
//            if( p.x>0 )
//            {
//                double w = leftW.at<double>(p);
//                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
//            }
//            if( p.x>0 && p.y>0 )
//            {
//                double w = upleftW.at<double>(p);
//                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
//            }
//            if( p.y>0 )
//            {
//                double w = upW.at<double>(p);
//                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
//            }
//            if( p.x<img.cols-1 && p.y>0 )
//            {
//                double w = uprightW.at<double>(p);
//                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
//            }
//        }
//    }
//}
//
///*
//  Estimate segmentation using MaxFlow algorithm
//*/
//static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
//{
//    graph.maxFlow();
//    Point p;
//    for( p.y = 0; p.y < mask.rows; p.y++ )
//    {
//        for( p.x = 0; p.x < mask.cols; p.x++ )
//        {
//            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
//            {
//                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
//                    mask.at<uchar>(p) = GC_PR_FGD;
//                else
//                    mask.at<uchar>(p) = GC_PR_BGD;
//            }
//        }
//    }
//}
//
//void cv::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
//                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
//                  int iterCount, int mode )
//{
//    Mat img = _img.getMat();
//    Mat& mask = _mask.getMatRef();
//    Mat& bgdModel = _bgdModel.getMatRef();
//    Mat& fgdModel = _fgdModel.getMatRef();
//
//    if( img.empty() )
//        CV_Error( CV_StsBadArg, "image is empty" );
//    if( img.type() != CV_8UC3 )
//        CV_Error( CV_StsBadArg, "image mush have CV_8UC3 type" );
//
//    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
//    Mat compIdxs( img.size(), CV_32SC1 );
//
//    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
//    {
//        if( mode == GC_INIT_WITH_RECT )
//            initMaskWithRect( mask, img.size(), rect );
//        else // flag == GC_INIT_WITH_MASK
//            checkMask( img, mask );
//        initGMMs( img, mask, bgdGMM, fgdGMM );
//    }
//
//    if( iterCount <= 0)
//        return;
//
//    if( mode == GC_EVAL )
//        checkMask( img, mask );
//
//    const double gamma = 50;
//    const double lambda = 9*gamma;
//    const double beta = calcBeta( img );
//
//    Mat leftW, upleftW, upW, uprightW;
//    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
//
//    for( int i = 0; i < iterCount; i++ )
//    {
//        GCGraph<double> graph;
//        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
//        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
//        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
//        estimateSegmentation( graph, mask );
//    }
//}







//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include <iostream>
//#include <time.h>
//
//using namespace std;
//using namespace cv;
//
//void help()
//{
//    cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
//    		"and then grabcut will attempt to segment it out.\n"
//    		"Call:\n"
//    		"./grabcut <image_name>\n"
//    	"\nSelect a rectangular area around the object you want to segment\n" <<
//        "\nHot keys: \n"
//        "\tESC - quit the program\n"
//        "\tr - restore the original image\n"
//        "\tn - next iteration\n"
//        "\n"
//        "\tleft mouse button - set rectangle\n"
//        "\n"
//        "\tCTRL+left mouse button - set GC_BGD pixels\n"
//        "\tSHIFT+left mouse button - set CG_FGD pixels\n"
//        "\n"
//        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
//        "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
//}
//
//const Scalar RED = Scalar(0,0,255);
//const Scalar PINK = Scalar(230,130,255);
//const Scalar BLUE = Scalar(255,0,0);
//const Scalar LIGHTBLUE = Scalar(255,255,160);
//const Scalar GREEN = Scalar(0,255,0);
//
//const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
//const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;
//
//void getBinMask( const Mat& comMask, Mat& binMask )
//{
//    if( comMask.empty() || comMask.type()!=CV_8UC1 )
//        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
//    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
//        binMask.create( comMask.size(), CV_8UC1 );
//    binMask = comMask & 1;
//}
//
//class GCApplication
//{
//public:
//    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
//    static const int radius = 2;
//    static const int thickness = -1;
//
//    void reset();
//    void setImageAndWinName( const Mat& _image, const string& _winName );
//    void showImage() const;
//    void mouseClick( int event, int x, int y, int flags, void* param );
//    int nextIter();
//    int getIterCount() const { return iterCount; }
//private:
//    void setRectInMask();
//    void setLblsInMask( int flags, Point p, bool isPr );
//
//    const string* winName;
//    const Mat* image;
//    Mat mask;
//    Mat bgdModel, fgdModel;
//
//    uchar rectState, lblsState, prLblsState;
//    bool isInitialized;
//
//    Rect rect;
//    vector<Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
//    int iterCount;
//};
//
//void GCApplication::reset()
//{
//    if( !mask.empty() )
//        mask.setTo(Scalar::all(GC_BGD));
//    bgdPxls.clear(); fgdPxls.clear();
//    prBgdPxls.clear();  prFgdPxls.clear();
//
//    isInitialized = false;
//    rectState = NOT_SET;
//    lblsState = NOT_SET;
//    prLblsState = NOT_SET;
//    iterCount = 0;
//}
//
//void GCApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
//{
//    if( _image.empty() || _winName.empty() )
//        return;
//    image = &_image;
//    winName = &_winName;
//    mask.create( image->size(), CV_8UC1);
//    reset();
//}
//
//void GCApplication::showImage() const
//{
//    if( image->empty() || winName->empty() )
//        return;
//
//    Mat res;
//    Mat binMask;
//    if( !isInitialized )
//        image->copyTo( res );
//    else
//    {
//        getBinMask( mask, binMask );
//        image->copyTo( res, binMask );
//    }
//
//    vector<Point>::const_iterator it;
//    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
//        circle( res, *it, radius, BLUE, thickness );
//    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
//        circle( res, *it, radius, RED, thickness );
//    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
//        circle( res, *it, radius, LIGHTBLUE, thickness );
//    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
//        circle( res, *it, radius, PINK, thickness );
//
//    if( rectState == IN_PROCESS || rectState == SET )
//        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);
//
//    imshow( *winName, res );
//}
//
//void GCApplication::setRectInMask()
//{
//    assert( !mask.empty() );
//    mask.setTo( GC_BGD );
//    rect.x = max(0, rect.x);
//    rect.y = max(0, rect.y);
//    rect.width = min(rect.width, image->cols-rect.x);
//    rect.height = min(rect.height, image->rows-rect.y);
//    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
//}
//
//void GCApplication::setLblsInMask( int flags, Point p, bool isPr )
//{
//    vector<Point> *bpxls, *fpxls;
//    uchar bvalue, fvalue;
//    if( !isPr )
//    {
//        bpxls = &bgdPxls;
//        fpxls = &fgdPxls;
//        bvalue = GC_BGD;
//        fvalue = GC_FGD;
//    }
//    else
//    {
//        bpxls = &prBgdPxls;
//        fpxls = &prFgdPxls;
//        bvalue = GC_PR_BGD;
//        fvalue = GC_PR_FGD;
//    }
//    if( flags & BGD_KEY )
//    {
//        bpxls->push_back(p);
//        circle( mask, p, radius, bvalue, thickness );
//    }
//    if( flags & FGD_KEY )
//    {
//        fpxls->push_back(p);
//        circle( mask, p, radius, fvalue, thickness );
//    }
//}
//
//void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
//{
//    // TODO add bad args check
//    switch( event )
//    {
//    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
//        {
//            bool isb = (flags & BGD_KEY) != 0,
//                 isf = (flags & FGD_KEY) != 0;
//            if( rectState == NOT_SET && !isb && !isf )
//            {
//                rectState = IN_PROCESS;
//                rect = Rect( x, y, 1, 1 );
//            }
//            if ( (isb || isf) && rectState == SET )
//                lblsState = IN_PROCESS;
//        }
//        break;
//    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
//        {
//            bool isb = (flags & BGD_KEY) != 0,
//                 isf = (flags & FGD_KEY) != 0;
//            if ( (isb || isf) && rectState == SET )
//                prLblsState = IN_PROCESS;
//        }
//        break;
//    case CV_EVENT_LBUTTONUP:
//        if( rectState == IN_PROCESS )
//        {
//            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
//            rectState = SET;
//            setRectInMask();
//            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
//            showImage();
//        }
//        if( lblsState == IN_PROCESS )
//        {
//            setLblsInMask(flags, Point(x,y), false);
//            lblsState = SET;
//            showImage();
//        }
//        break;
//    case CV_EVENT_RBUTTONUP:
//        if( prLblsState == IN_PROCESS )
//        {
//            setLblsInMask(flags, Point(x,y), true);
//            prLblsState = SET;
//            showImage();
//        }
//        break;
//    case CV_EVENT_MOUSEMOVE:
//        if( rectState == IN_PROCESS )
//        {
//            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
//            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
//            showImage();
//        }
//        else if( lblsState == IN_PROCESS )
//        {
//            setLblsInMask(flags, Point(x,y), false);
//            showImage();
//        }
//        else if( prLblsState == IN_PROCESS )
//        {
//            setLblsInMask(flags, Point(x,y), true);
//            showImage();
//        }
//        break;
//    }
//}
//
//int GCApplication::nextIter()
//{
//    if( isInitialized )
//        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
//    else
//    {
//        if( rectState != SET )
//            return iterCount;
//
//        if( lblsState == SET || prLblsState == SET )
//            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
//        else
//            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );
//
//        isInitialized = true;
//    }
//    iterCount++;
//
//    bgdPxls.clear(); fgdPxls.clear();
//    prBgdPxls.clear(); prFgdPxls.clear();
//
//    return iterCount;
//}
//
//GCApplication gcapp;
//
//void on_mouse( int event, int x, int y, int flags, void* param )
//{
//    gcapp.mouseClick( event, x, y, flags, param );
//}
//
//int main( int argc, char** argv )
//{
//    if( argc!=2 )
//    {
//    	help();
//        return 1;
//    }
//    string filename = argv[1];
//    if( filename.empty() )
//    {
//    	cout << "\nDurn, couldn't read in " << argv[1] << endl;
//        return 1;
//    }
//    Mat image = imread( filename, 1 );
//    if( image.empty() )
//    {
//        cout << "\n Durn, couldn't read image filename " << filename << endl;
//    	return 1;
//    }
//
//    help();
//
//    const string winName = "image";
//    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
//    cvSetMouseCallback( winName.c_str(), on_mouse, 0 );
//
//    gcapp.setImageAndWinName( image, winName );
//    gcapp.showImage();
//
//    for(;;)
//    {
//        int c = cvWaitKey(0);
//        switch( (char) c )
//        {
//        case '\x1b':
//            cout << "Exiting ..." << endl;
//            goto exit_main;
//        case 'r':
//            cout << endl;
//            gcapp.reset();
//            gcapp.showImage();
//            break;
//        case 'n':
//			clock_t time_cal=0;
//			time_cal=clock();
//            int iterCount = gcapp.getIterCount();
//            cout << "<" << iterCount << "... ";
//            int newIterCount = gcapp.nextIter();
//            if( newIterCount > iterCount )
//            {
//                gcapp.showImage();
//                cout << iterCount << ">" << endl;
//            }
//            else
//                cout << "rect must be determined>" << endl;
//			time_cal=clock()-time_cal;
//			cout<<time_cal<<"ms"<<endl;
//            break;
//        }
//    }
//
//exit_main:
//    cvDestroyWindow( winName.c_str() );
//    return 0;
//}

