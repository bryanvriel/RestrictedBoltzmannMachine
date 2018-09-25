//-*- C++ -*-

#include <cstdio>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "RBM.h"

using namespace std;

double diffclock(clock_t clock1,clock_t clock2) {
    double diffticks = clock1 - clock2;
    double diffms = diffticks / CLOCKS_PER_SEC;
    return diffms;
}


int main() {

    char outputbuf[50];

    const string filename = "data_full/all.dta.bin.zeroProbe";
    const long N_movies = 17770;
    const long N_feature = 100;
    int data_buffer[4] = {0, 0, 0, 0};
    const size_t recl = 4 * sizeof(int);

    const long Korder = 5;
    
    // Instantiate an RBM and allocate enough memory
    RBMcond rbm(N_feature, N_movies, Korder);
    rbm.allocate();

    // --------------------------------------------------------------------------------
    // Read in the ratings on a per-user basis
    // --------------------------------------------------------------------------------

    printf("\nReading in the training data by users\n");
    ifstream ifid;
    ifid.open(filename.c_str(), ios::in | ios::binary);

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

    // --------------------------------------------------------------------------------
    // Train the RBM
    // --------------------------------------------------------------------------------
    
    // Initialize the biases
    rbm.initBiases(usersTrain);

    // Stochastic gradient descent - loop over the ratings
    size_t numEpochs =  60;                  // number of training epochs
    unsigned short numCD = 1;               // number of CD steps
    double lRate_factor = 1.0;
    float momentum;
    size_t nproc = 12;

    // Coarse-grained epochs
    printf("Training the RBM\n");
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {

        // Shuffle the data
        random_shuffle(usersTrain.begin(), usersTrain.end());

        // Adaptive learning rate
        //lRate_factor = 1.0 / ((double) epoch + 1.0);
        //lRate_factor = 1.0;

        if (epoch < 15)
            numCD = 1;
        else if ((epoch >= 15) && (epoch < 30))
            numCD = 3;
        else
            numCD = 5;

        if (epoch < 5)
            momentum = 0.8;
        else
            momentum = 0.9;

        printf(" - learning rate factor: %f numCD: %d\n", lRate_factor, numCD);

        // Run one epoch
        time_t tstart = time(0);
        thread_train(&rbm, usersTrain, lRate_factor, numCD, momentum, nproc);
        time_t tend = time(0);
        printf(" - time for epoch: %zd s\n", tend - tstart);

        // Save the state for this epoch
        //sprintf(outputbuf, "states_cond_%03zdfeat_May03/savedRBM_%03zu.h5", N_feature, epoch);
        sprintf(outputbuf, "states_cond_%03zdfeat_May05A/savedRBM_%03zu.h5", N_feature, epoch);
        rbm.saveState(outputbuf);

        // Adjust the learning rate
        lRate_factor *= 0.9;

    }

    usersTrain.clear();
    return 0;

}

// end of file
