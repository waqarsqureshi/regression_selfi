#include <iostream>
#include <vector>
#include <dlib/svm.h>
#include <dlib/matrix.h>

#include <text/csv/ostream.hpp>    
#include "text/csv/istream.hpp"

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
#include <sstream>  
#include <string>

using namespace dlib;
using namespace std;
/*
            REQUIREMENTS ON K 
                is a kernel function object as defined in dlib/svm/kernel_abstract.h 

            INITIAL VALUE
                - get_lambda() == 0
                - basis_loaded() == false
                - get_max_basis_size() == 400
                - will_use_regression_loss_for_loo_cv() == true
                - get_search_lambdas() == logspace(-9, 2, 50) 
                - this object will not be verbose unless be_verbose() is called

            WHAT THIS OBJECT REPRESENTS
                This object represents a tool for performing kernel ridge regression 
                (This basic algorithm is also known my many other names, e.g. regularized 
                least squares or least squares SVM). 

                The exact definition of what this algorithm does is this:
                    Find w and b that minimizes the following (x_i are input samples and y_i are target values):
                        lambda*dot(w,w) + sum_over_i( (f(x_i) - y_i)^2 )
                        where f(x) == dot(x,w) - b

                    Except the dot products are replaced by kernel functions.  So this
                    algorithm is just regular old least squares regression but with the
                    addition of a regularization term which encourages small w and the
                    application of the kernel trick.


                It is implemented using the empirical_kernel_map and thus allows you 
                to run the algorithm on large datasets and obtain sparse outputs.  It is also
                capable of estimating the lambda parameter using leave-one-out cross-validation.


                The leave-one-out cross-validation implementation is based on the techniques
                discussed in this paper:
                    Notes on Regularized Least Squares by Ryan M. Rifkin and Ross A. Lippert.
        !*/


// This is the object of csv reader
//==============================
namespace csv = ::text::csv;   
//==============================
// the input parameters are
// face height/(body height+ face height) = 
// shoulder slope with the horizantal     = H  1-S(0)
// shoulder width/body height             = I  2-S(1)
// normalized shoulder width              = J  3-S(2)
// normalized shoulder width              = 
// face height/body height                = K  4-S(3)
// face height/shoulder width             = L  5-S(4)


typedef matrix<double,5,1> input_sample;// sample measurement size is 5 samples as shown above
typedef double posX;// output 1
typedef double posY;// output 2
typedef double posZ;// output 3
typedef double zoom;// we keep zoom constant and does not include as output
typedef double pitch;// output 4
typedef double yaw;// output 5

int main()
{
        posX x;posY y; posZ z; zoom ZOOM;pitch PITCH; yaw YAW; input_sample S;
        std::vector<double> posXVector;
        std::vector<double> posYVector;
        std::vector<double> posZVector;
        std::vector<double> zoomVector;
        std::vector<double> pitchVector;
        std::vector<double> yawVector;
        std::vector<input_sample> samples;
        
        
        // get input data for training by reading the data from file
         std::ifstream fs("/home/user/regression_selfi/build/output-10k.csv");
         csv::csv_istream csvs(fs);
         // get the labels first from the csv file
         std::string imageName,posXLabel,posYLabel,posZLabel,pitchLabel,yawLabel,zoomLabel;
         std::string sampleLabelH, sampleLabelI,sampleLabelJ, sampleLabelK, sampleLabelL;
         // read from the file
         csvs>>imageName>>posXLabel>>posYLabel>>posZLabel>>zoomLabel>>pitchLabel>>yawLabel;
         csvs>>sampleLabelH>>sampleLabelI>>sampleLabelJ>>sampleLabelK>>sampleLabelL;
         // debug print start
         std::cout<<imageName<<" : "<<posXLabel<<" : "<<posYLabel<<" : "<<posZLabel<<" : "<<zoomLabel<<" : "<<pitchLabel<<" : "<<yawLabel<<":::";
         std::cout<<sampleLabelH<<" : "<<sampleLabelI<<" : "<<sampleLabelJ<<" : "<<sampleLabelK<<" : "<<sampleLabelL<<" : "<<std::endl;
         // debug print ends

         while (csvs) {
                csvs>> imageName;// no need to save it anywhere column A
                csvs>>x>>y>>z>>ZOOM>>PITCH>>YAW;// column B,C,D,E,F,G;
                // add the sample data
                double temp;
                csvs>>temp;//H
                S(0,0) = temp;
                csvs>>temp;//I
                S(1,0) = temp;
                csvs>>temp;//J
                S(2,0) = temp;
                csvs>>temp;//K
                S(3,0) = temp;
                csvs>>temp; //L
                S(4,0) = temp;
                // save into the vectors            
                posXVector.push_back(x);posYVector.push_back(y);posZVector.push_back(z);
                zoomVector.push_back(ZOOM);pitchVector.push_back(PITCH);yawVector.push_back(YAW);
                // save sample into the vector
                samples.push_back(S);
                // debug print values
                //std::cout<<imageName<<":"<<x<<":"<<y<<":"<<z<<":"<<ZOOM<<":"<<PITCH<<":"<<YAW<<":"<<S<<std::endl;
         }

        // Now we are making a typedef for the kind of kernel we want to use.  I picked the
        // radial basis kernel 
        typedef radial_basis_kernel<input_sample> kernel_type;
        krr_trainer<kernel_type> trainer;
        const double gamma = 3.0/compute_mean_squared_distance(randomly_subsample(samples, 2000));// 2000 samples
        trainer.set_kernel(kernel_type(gamma));
        
        // train the regressors
        // now train a function based on our sample points
        decision_function<kernel_type> test_posX = trainer.train(samples, posXVector);
        decision_function<kernel_type> test_posY = trainer.train(samples, posYVector);
        decision_function<kernel_type> test_posZ = trainer.train(samples, posZVector);
        decision_function<kernel_type> test_pitch = trainer.train(samples, pitchVector);
        decision_function<kernel_type> test_yaw = trainer.train(samples, yawVector);
        
        //serialize("saved_function.dat") << test_posX<<test_posY<<test_pitch<<test_yaw;
        serialize("saved_function.dat") << test_posZ<< test_posY << test_posX << test_pitch << test_yaw;
        deserialize("saved_function.dat") >> test_posZ>> test_posY >> test_posX >> test_pitch >> test_yaw;
        S(0,0) =-0.056846;
        S(1,0)=0.289979;
        S(2,0) =0.10625;	
        S(3,0) =0.172708;	
        S(4,0) =0.595588;
//Image_2601.jpg	25.8	123.755	37.8	94	-18.6	82.1999

        std::cout<<test_posX(S)<<" : ";
        std::cout<<test_posY(S)<<" : ";
        std::cout<<test_posZ(S)<<" : ";
        std::cout<<test_pitch(S)<<" : ";
        std::cout<<test_yaw(S)<<" : ";
        std::cout<<std::endl;

        
        // product the value fom regressors
        
      std::vector<double> loo_values_x, loo_values_y,loo_values_z,loo_values_pitch,loo_values_yaw;
      test_posX = trainer.train(samples, posXVector, loo_values_x);
      test_posY = trainer.train(samples, posYVector, loo_values_y);
      test_posZ = trainer.train(samples, posZVector, loo_values_z);
      test_pitch = trainer.train(samples, pitchVector, loo_values_pitch);
      test_yaw = trainer.train(samples, yawVector, loo_values_yaw);
      
      
      cout << "mean squared LOO error X: " << mean_squared_error(posXVector,loo_values_x) << endl;
      cout << "R^2 LOO value X:          " << r_squared(posXVector,loo_values_x) << endl;
      
      cout << "mean squared LOO error Y: " << mean_squared_error(posYVector,loo_values_y) << endl;
      cout << "R^2 LOO value Y:          " << r_squared(posYVector,loo_values_y) << endl;
      
      cout << "mean squared LOO error Z: " << mean_squared_error(posZVector,loo_values_z) << endl;
      cout << "R^2 LOO value Z:          " << r_squared(posZVector,loo_values_z) << endl;
      
      cout << "mean squared LOO error pitch: " << mean_squared_error(pitchVector,loo_values_pitch) << endl;
      cout << "R^2 LOO value pitch:          " << r_squared(pitchVector,loo_values_pitch) << endl;
      
      cout << "mean squared LOO error yaw: " << mean_squared_error(yawVector,loo_values_yaw) << endl;
      cout << "R^2 LOO value yaw:          " << r_squared(yawVector,loo_values_yaw) << endl;
      
      
      serialize("saved_function_10k.dat") << test_posZ<< test_posY << test_posX << test_pitch << test_yaw;
        //deserialize("saved_function.dat") >> test_posZ>> test_posY >> test_posX >> test_pitch >> test_yaw;
}


