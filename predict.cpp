//-*- C++ -*-

#include <cstdio>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "RBM.h"

using namespace std;

double diffclock(clock_t clock1,clock_t clock2) {
    double diffticks = clock1 - clock2;
    double diffms = diffticks / CLOCKS_PER_SEC;
    return diffms;
}


int main(int argc, char * argv[]) {

    if (argc < 4) {
        printf("\n");
        printf("Usage: predict.cpp state.h5 data.dta.bin predictions.txt\n\n");
        exit(1);
    }
    const string trainfile = "data_full/training.dta.bin";
    const string h5file(argv[1]);
    const string testfile(argv[2]);
    const string outputfile(argv[3]);
    const long N_movies = 17770;
    const long N_feature = 100;
    const long Korder = 5;
    const size_t recl = 4 * sizeof(int);
    int data_buffer[4] = {0, 0, 0, 0};
    
    // Instantiate an RBM and allocate enough memory
    RBM rbm(N_feature, N_movies, Korder);
    rbm.allocate();

    // --------------------------------------------------------------------------------
    // Different strategy for reading in the data - we read primarily by user.
    // --------------------------------------------------------------------------------

    ifstream ifid;
    ifid.open(trainfile, ios::in | ios::binary);

    // Training - read the first line
    vector<user_t> usersTrain;
    user_t user;
    ifid.read((char *) &data_buffer, recl);
    user.userId = data_buffer[0];
    user.movieIds.push_back(data_buffer[1]);
    user.ratings.push_back(data_buffer[3]);

    // Read the rest of the lines
    while (ifid.read((char *) &data_buffer, recl)) { 
        int userId = data_buffer[0];
        if (userId != user.userId) {
            // Store the current user
            usersTrain.push_back(user);
            // Reset the data
            user.userId = userId;
            user.movieIds.clear();
            user.ratings.clear();
            user.movieIds.push_back(data_buffer[1]);
            user.ratings.push_back(data_buffer[3]);
        }
        else {
            user.movieIds.push_back(data_buffer[1]);
            user.ratings.push_back(data_buffer[3]);
        }
    }

    // Save the final user
    usersTrain.push_back(user);
    ifid.close();
    printf("Number of training users: %zu\n", usersTrain.size());

    // Consistency check - make sure that we have training data for all users
    printf("Checking consistency\n");
    for (size_t i = 0; i < usersTrain.size(); ++i) {
        if (int(i) != usersTrain[i].userId) {
            printf("Something wrong\n");
            exit(2);
        }
    }
    printf(" - Passed\n");

    // Testing - read the first line
    vector<user_t> usersTest;
    ifid.open(testfile, ios::in | ios::binary);
    user.movieIds.clear(); user.ratings.clear();
    ifid.read((char *) &data_buffer, recl);
    user.userId = data_buffer[0];
    user.movieIds.push_back(data_buffer[1]);
    user.ratings.push_back(data_buffer[3]);

    // Read the rest of the lines
    size_t numTest = 1;
    while (ifid.read((char *) &data_buffer, recl)) { 
        int userId = data_buffer[0];
        if (userId != user.userId) {
            // Store the current user
            usersTest.push_back(user);
            // Reset the data
            user.userId = userId;
            user.movieIds.clear();
            user.ratings.clear();
            user.movieIds.push_back(data_buffer[1]);
            user.ratings.push_back(data_buffer[3]);
        }
        else {
            user.movieIds.push_back(data_buffer[1]);
            user.ratings.push_back(data_buffer[3]);
        }
        ++numTest;
    }

    // Save the final user
    usersTest.push_back(user);
    printf("Number of testing points: %zu\n", numTest);

    // Allocate a vector of floats to hold the predicted rating
    vector<float> predictedRatings(numTest,0.0);

    // --------------------------------------------------------------------------------
    // Load the RBM state and generate predictions.
    // --------------------------------------------------------------------------------
    
    // Let the RBM load a prior state
    printf("Loading the H5 state\n");
    rbm.loadState(h5file);

    // Generate predictions for the testing data
    printf("Predicting ratings\n");
    clock_t tstart = clock();
    rbm.predict(usersTrain, usersTest, predictedRatings);
    clock_t tend = clock();
    printf(" - time for predicting: %f s\n", double(diffclock(tend,tstart)));

    // Write ratings to file
    FILE *fout = fopen(outputfile.c_str(), "w");
    for (vector<float>::iterator it = predictedRatings.begin(); 
            it != predictedRatings.end(); ++it){
        fprintf(fout, "%5.3f\n", *it);
    }
    fclose(fout);
    
    usersTrain.clear();
    usersTest.clear();
    predictedRatings.clear();
    return 0;

}


// end of file
