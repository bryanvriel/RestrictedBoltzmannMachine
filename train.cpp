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

    const string filename = "data_full/training.dta.bin";
    const long N_movies = 17770;
    const long N_feature = 100;
    int data_buffer[4] = {0, 0, 0, 0};
    const size_t recl = 4 * sizeof(int);

    const long Korder = 5;
    
    // Instantiate an RBM and allocate enough memory
    RBM rbm(N_feature, N_movies, Korder);
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

    // Shuffle the data
    printf("Shuffling the data\n");
    random_shuffle(usersTrain.begin(), usersTrain.end());

    // Initialize the biases
    rbm.initBiases(usersTrain);

    // Stochastic gradient descent - loop over the ratings
    size_t numEpochs = 50;                  // number of training epochs
    float learningRate = 0.0012;            // learning rate
    unsigned short numCD = 1;               // number of Contrastive Divergence steps
    float wPenalty = 0.00025;               // Weight decay regularization parameters

    // Coarse-grained epochs
    printf("Training the RBM\n");
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {

        // Run one epoch
        clock_t tstart = clock();
        rbm.train(usersTrain, learningRate, numCD, wPenalty);
        clock_t tend = clock();
        printf(" - time for epoch: %f s\n", double(diffclock(tend,tstart)));

        // Save the state for this epoch
        sprintf(outputbuf, "states/savedRBM_%03zu.h5", epoch);
        rbm.saveState(outputbuf);

    }

    usersTrain.clear();
    return 0;

}


// end of file
