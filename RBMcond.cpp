// -*- C++ -*-

#include <functional>
#include <algorithm>
#include <random>
#include <pthread.h>
#include "H5Cpp.h"
#include "RBM.h"

// Parameters for BigChaos 2008, RBMV3, model 24 (100 features)
#define epsilonW            0.005
#define epsilonD            0.0005
#define epsilonVb           0.005
#define epsilonHb           0.005
#define wgtPenalty          0.0005


static void * train_wrapper(void * arg) {
    /*
    C-type wrapper for RBM::train class method.
    */

    // Unpack the context and arguments
    context * ctxt = (context *) arg;
    std::vector<user_t> & users = ctxt->users;
    double lRate_factor = ctxt->info->lRate_factor;
    unsigned short nCD = ctxt->info->nCD;
    float momentum = ctxt->info->momentum;

    // Call RBM training
    ctxt->info->rbm->train(users, lRate_factor, nCD, momentum, &ctxt->info->mutex);

    // Done
    pthread_exit(0);
    return NULL;
}


RBMcond::RBMcond(size_t nhidden, size_t nvisible, size_t korder) : 
    RBM(nhidden, nvisible, korder) {
    /*
    Constructor
    */
}


void RBMcond::allocate() {
    /*
    Allocate memory for weighting matrix and user and hidden biases.
    */

    // Initialize the random number generator
    _Trand = gsl_rng_mt19937;
    _rng = gsl_rng_alloc(_Trand);
    
    // Allocate weight matrix with random numbers
    _W.allocate(_numVisible*_K, _numHidden);
    _W.random(0.0, 0.01);
    
    // Allocate biases with zeros
    _hbias.resize(_numHidden);
    std::fill(_hbias.begin(), _hbias.end(), 0.0);

    _vbias.allocate(_numVisible, _K);
    _vbias.zeros();
    
    // Allocate D matrix with random numbers
    _D.allocate(_numVisible, _numHidden);
    _D.random(0.0, 0.0005);

    // Movie count vector
    _movieRcount.resize(_numVisible*_K, 0);

    // Allocate the incremental matrices
    _dW.allocate(_numVisible*_K, _numHidden);
    _dVbias.allocate(_numVisible, _K);
    _dHbias.resize(_numHidden);
    _dD.allocate(_numVisible, _numHidden);

    _dW.zeros();
    _dVbias.zeros();
    _dD.zeros();
    std::fill(_dHbias.begin(), _dHbias.end(), 0.0);

}


void thread_train(RBMcond * rbm, std::vector<user_t> & allUsers, double lRate_factor,
    unsigned short nCD, float momentum, size_t nproc) {
    /*
    Pthread function to train over my subset of the users.
    */
    size_t nusers = allUsers.size();
    size_t nominal_load = nusers / nproc;

    // Initialize context structure for pthreads
    common info;
    info.nproc = nproc;
    info.lRate_factor = lRate_factor;
    info.nCD = nCD;
    info.rbm = rbm;

    // Initialize pthread problem
    pthread_mutex_init(&info.mutex, 0);
    pthread_t threads[nproc];

    // Create array of context structures
    context thread_context[nproc];

    size_t istart = 0;
    size_t procN;
    //printf("Launching threads\n");
    for (size_t cnt = 0; cnt < nproc; ++cnt) {

        if (cnt == nproc - 1)
            procN = nusers - cnt * nominal_load;
        else
            procN = nominal_load;
        //printf("Thread %zu numUsers = %zu\n", cnt, procN);

        // Point to common problem info
        thread_context[cnt].info = &info;

        // Subset the users
        std::vector<user_t>::const_iterator first = allUsers.begin() + istart;
        std::vector<user_t>::const_iterator last = allUsers.begin() + istart + procN;
        std::vector<user_t> subUsers(first, last);
        thread_context[cnt].users.reserve(procN);
        thread_context[cnt].users = subUsers;

        // Launch training
        int status = pthread_create(&threads[cnt], 0, train_wrapper, (void *) &thread_context[cnt]);
        if (status)
            printf("Error %d in pthread_create\n", status);

        istart += procN;
    }

    // Wait for threads to finish
    for (size_t cnt = 0; cnt < nproc; ++cnt) {
        pthread_join(threads[cnt], 0);
    }

}


void RBMcond::train(std::vector<user_t> & users, double lRate_factor, unsigned short nCD,
    float momentum, pthread_mutex_t * mutex) {
    /*
    Train the RBM given a list of user structurs.
    */
    size_t nusers = users.size();

    // Initialize a uniform random number generator for hidden states
    std::random_device rd;
    std::mt19937 randGen(rd());
    std::uniform_real_distribution<float> randDist(0.0, 1.0);

    // Initialize some working vectors and variables
    vector_t hidden_state(_numHidden,0.0);
    vector_t pos_hidden_probs(_numHidden,0.0);
    std::vector<double> temp_visible_prob(_K,0.0);
    vector_t visible_units;
    vector_t vWsum(_numHidden,0.0);

    // Loop over the users
    float sumSquareError = 0.0;
    size_t totalRatings = 0;
    for (size_t uIndex = 0; uIndex < nusers; ++uIndex) {

        // Get the current user
        user_t user = users[uIndex];
        size_t nTotalRatings = user.ratings.size();
        size_t nTrainRatings = 0;

        // Seed the Gibbs sampling with the training set to compute the hidden probabilities
        std::fill(vWsum.begin(), vWsum.end(), 0.0);
        for (size_t i = 0; i < nTotalRatings; ++i) {
            int rating = user.ratings[i];
            int visibleIndex = user.movieIds[i];
            // If non-zero rating, add to vWsum
            if (rating > 0) {
                for (size_t j = 0; j < _numHidden; ++j) {
                    vWsum[j] += _W(visibleIndex*_K + rating - 1,j);
                }
                ++nTrainRatings;
            }
            // Conditional contribution
            for (size_t j = 0; j < _numHidden; ++j) {
                vWsum[j] += _D(visibleIndex,j);
            }
        }

        // Compute probabilities and binary states
        for (size_t j = 0; j < _numHidden; ++j) {
            float binaryState = _sigmoid(_hbias[j] + vWsum[j]) > randDist(randGen);
            hidden_state[j] = binaryState;
        }
        pos_hidden_probs = hidden_state;

        // Allocate memory for sampled visible units
        visible_units.resize(nTrainRatings * _K);

        // Now do rest of CD steps
        for (unsigned short T = 0; T < nCD; ++T) {

            // Sample the visible units given the hidden units
            size_t traincount = 0;
            for (size_t i = 0; i < nTotalRatings; ++i) {
                int rating = user.ratings[i];
                if (rating > 0) {
                    size_t visibleIndex = user.movieIds[i];
                    size_t offset = visibleIndex * _K;
                    double sumProbability = 0.0;
                    for (size_t k = 0; k < _K; ++k ) {
                        float hWsum = 0.0;
                        for (size_t j = 0; j < _numHidden; ++j) {
                            hWsum += hidden_state[j] * _W(offset+k,j);
                        }
                        double expActivation = exp(_vbias(visibleIndex,k) + hWsum);
                        temp_visible_prob[k] = expActivation;
                        sumProbability += expActivation;
                    }

                    // Normalize the probabilities and copy to main visible unit
                    for (size_t k = 0; k < _K; ++k) {
                        temp_visible_prob[k] /= sumProbability;
                    }
                    visible_units[traincount*_K  ] = temp_visible_prob[0];
                    visible_units[traincount*_K+1] = temp_visible_prob[1];
                    visible_units[traincount*_K+2] = temp_visible_prob[2];
                    visible_units[traincount*_K+3] = temp_visible_prob[3];
                    visible_units[traincount*_K+4] = temp_visible_prob[4];
                    ++traincount;
 
                }                
            }

            // Sample the hidden units given the visible units
            std::fill(vWsum.begin(), vWsum.end(), 0.0);
            traincount = 0;
            for (size_t i = 0; i < nTotalRatings; ++i) {
                int rating = user.ratings[i];
                int visibleIndex = user.movieIds[i];
                // If non-zero rating, add visible unit contribution to vWsum
                if (rating > 0) {
                    int offset = visibleIndex * _K;
                    float vu1 = visible_units[traincount*_K];
                    float vu2 = visible_units[traincount*_K+1];
                    float vu3 = visible_units[traincount*_K+2];
                    float vu4 = visible_units[traincount*_K+3];
                    float vu5 = visible_units[traincount*_K+4];
                    for (size_t j = 0; j < _numHidden; ++j) {
                        vWsum[j] += vu1 * _W(offset,  j) + vu2 * _W(offset+1,j)
                                  + vu3 * _W(offset+2,j) + vu4 * _W(offset+3,j)
                                  + vu5 * _W(offset+4,j);
                    }
                    ++traincount;
                }
                // Conditional contribution
                for (size_t j = 0; j < _numHidden; ++j) {
                    vWsum[j] += _D(visibleIndex,j);
                }
            }

            // Compute probabilities and binary states
            if (T == (nCD - 1)) {
                // Need only probabilities for last CD step
                for (size_t j = 0; j < _numHidden; ++j) {
                    hidden_state[j] = _sigmoid(_hbias[j] + vWsum[j]);
                }
            }
            else {
                // Get binary states for all other CD steps
                for (size_t j = 0; j < _numHidden; ++j) {
                    hidden_state[j] = _sigmoid(_hbias[j] + vWsum[j]) > randDist(randGen);
                }
            }

        }

        // Update the weights and biases (lock the mutex)
        //pthread_mutex_lock(mutex);
        float userError = _update(user.ratings, user.movieIds, visible_units, pos_hidden_probs, 
            hidden_state, lRate_factor, momentum);
        //pthread_mutex_unlock(mutex);

        // Accumulate errors and count
        sumSquareError += userError;
        totalRatings += nTrainRatings;
        
    }

     // Print the current error
    printf(" - training RMSE: %f\n", sqrt(sumSquareError / float(totalRatings)));

    // Clear the vectors
    hidden_state.clear();
    pos_hidden_probs.clear();
    temp_visible_prob.clear();
    visible_units.clear();
    vWsum.clear();

}


float RBMcond::_update(std::vector<int> & ratings, std::vector<int> & movieIds, 
    std::vector<float> & visible_units, vector_t & pos_hidden_probs, 
    vector_t & neg_hidden_probs, double lRfact, float momentum) {
    /*
    Update the weight matrix and biases.
    */
    const size_t numTotalRatings = ratings.size();
    float error = 0.0;
    float v0, vk;
    size_t traincount = 0;
    for (size_t i = 0; i < numTotalRatings; ++i) {
        int r = ratings[i] - 1;
        size_t visibleIndex = movieIds[i];
        // Update W and vbias only if user provided a rating
        if (r > -1) {
            size_t offset = visibleIndex * _K;
            for (size_t k = 0; k < _K; ++k) {

                // Unpack the observed and predicted ratings
                if (int(k) == r)
                    v0 = 1.0;
                else
                    v0 = 0.0;
                vk = visible_units[traincount*_K + k];
               
                // Update the weights 
                for (size_t j = 0; j < _numHidden; ++j) {
                    // Compute gradient for W
                    float gradient = pos_hidden_probs[j] * v0 - neg_hidden_probs[j] * vk;
                    // Get old values
                    float dW = _dW(offset+k,j);
                    float Wold = _W(offset+k,j);
                    // Momentum-modulated gradient
                    dW = momentum * dW + lRfact * epsilonW * (gradient - wgtPenalty * Wold);
                    _W(offset+k,j) += dW;           
                    _dW(offset+k,j) = dW;
                }

                // Update the visible bias
                float deltaV = v0 - vk;
                float dVb = momentum * _dVbias(visibleIndex,k) + lRfact * epsilonVb * deltaV;
                _vbias(visibleIndex,k) += dVb;
                _dVbias(visibleIndex,k) = dVb;

                //// Accumulate the error
                error += deltaV * deltaV;
            }
            ++traincount;
        }

        // Update D matrix
        for (size_t j = 0; j < _numHidden; ++j) {
            float gradient = pos_hidden_probs[j] - neg_hidden_probs[j];
            float dD = momentum * _dD(visibleIndex,j) + lRfact * epsilonD * gradient;            
            _D(visibleIndex,j) += dD;
            _dD(visibleIndex,j) = dD;
        }

    }

    // Update hidden unit biases separately
    for (size_t j = 0; j < _numHidden; ++j) {
        float gradient = pos_hidden_probs[j] - neg_hidden_probs[j];
        float dHb = momentum * _dHbias[j] + lRfact * epsilonHb * gradient;
        _hbias[j] += dHb;
        _dHbias[j] = dHb;
    }

    return error;
}


void RBMcond::predict(std::vector<user_t> & training_users, std::vector<user_t> & testing_users,
                  vector_t & predictedRatings) {
    /*
    Use the trained RBM to predict ratings for users.
    */

    // Initialize some work vectors
    vector_t visible_prob(_K,0.0);
    vector_t hidden_prob(_numHidden,0.0);

    // Loop over the users in the testing vector
    size_t rcnt = 0;
    for (size_t n = 0; n < testing_users.size(); ++n) {

        // Get the testing user
        user_t user = testing_users[n];
        
        // Get this user's training data
        user_t user_train = training_users[user.userId];
        if (user_train.userId != user.userId) {
            printf("Training user does not match test user\n");
            exit(2);
        }
        size_t numTrain = user_train.ratings.size();

        // Get probabilities for hidden units given the training data
        for (size_t j = 0; j < _numHidden; ++j) {
            float hsum = 0.0;
            for (size_t i = 0; i < numTrain; ++i) {
                size_t rating = user_train.ratings[i];
                size_t visibleIndex = user_train.movieIds[i];
                if (rating > 0) {
                    hsum += _W(visibleIndex*_K + rating - 1, j);
                }
                hsum += _D(visibleIndex,j);
            }
            hidden_prob[j] = _sigmoid(_hbias[j] + hsum);
        }

        // Compute predicted rating
        for (size_t i = 0; i < user.movieIds.size(); ++i) {
            // Compute probabilities
            _sample_v_from_h(hidden_prob, visible_prob, user.movieIds[i]);
            // Compute the expectation
            float rating = 1.0 * visible_prob[0] + 2.0 * visible_prob[1]
                         + 3.0 * visible_prob[2] + 4.0 * visible_prob[3] 
                         + 5.0 * visible_prob[4];
            predictedRatings[rcnt] = rating;
            ++rcnt;
        }
    }

    visible_prob.clear();
    hidden_prob.clear();
    
}


void RBMcond::saveState(char * filename) const {
    /*
    Save the weight matrix and biases to an HDF5 file.
    */

    // Create the HDF5 file
    H5::H5File * h5file = new H5::H5File(filename, H5F_ACC_TRUNC);

    // Some HDF5 initialization
    hsize_t fdim3[3];
    hsize_t fdim2[2];
    hsize_t fdim1[1];
    H5::DataSpace dataspace;
    H5::DataSet dataset;

    // Write out W
    fdim3[0] = _numVisible;
    fdim3[1] = _K;
    fdim3[2] = _numHidden;
    dataspace = H5::DataSpace(3, fdim3);
    dataset = H5::DataSet(h5file->createDataSet("W", H5::PredType::NATIVE_FLOAT, dataspace));
    dataset.write(_W._data, H5::PredType::NATIVE_FLOAT, dataspace);

    // Write out hidden unit biases
    fdim1[0] = _numHidden;
    dataspace = H5::DataSpace(1, fdim1);
    dataset = H5::DataSet(h5file->createDataSet("hbias", H5::PredType::NATIVE_FLOAT, dataspace));
    dataset.write(&_hbias[0], H5::PredType::NATIVE_FLOAT, dataspace);

    // Write out visible unit biases
    fdim2[0] = _K;
    fdim2[1] = _numVisible;
    dataspace = H5::DataSpace(2, fdim2);
    dataset = H5::DataSet(h5file->createDataSet("vbias", H5::PredType::NATIVE_FLOAT, dataspace));
    dataset.write(_vbias._data, H5::PredType::NATIVE_FLOAT, dataspace);

    // Write out D
    fdim2[0] = _numVisible;
    fdim2[1] = _numHidden;
    dataspace = H5::DataSpace(2, fdim2);
    dataset = H5::DataSet(h5file->createDataSet("D", H5::PredType::NATIVE_FLOAT, dataspace));
    dataset.write(_D._data, H5::PredType::NATIVE_FLOAT, dataspace);

    // Clean HDF5
    delete h5file;

}


void RBMcond::loadState(const std::string filename) {
    /*
    Save the weight matrix and biases to an HDF5 file.
    */

    // Open the HDF5 file for reading
    H5::H5File * h5file = new H5::H5File(filename.c_str(), H5F_ACC_RDONLY);

    // Some HDF5 initialization
    hsize_t fdim3[3];
    hsize_t fdim2[2];
    hsize_t fdim1[1];
    H5::DataSpace dataspace;
    H5::DataSpace memspace;
    H5::DataSet dataset;

    // Set the dimensions
    fdim3[0] = _numVisible;
    fdim3[1] = _K;
    fdim3[2] = _numHidden;
    fdim2[0] = _K;
    fdim2[1] = _numVisible;
    fdim1[0] = _numHidden;

    // Read W
    dataset = H5::DataSet(h5file->openDataSet("W"));
    dataspace = dataset.getSpace();
    memspace = H5::DataSpace(3, fdim3);
    dataset.read(_W._data, H5::PredType::NATIVE_FLOAT, memspace, dataspace);

    // Read hidden unit biases
    dataset = H5::DataSet(h5file->openDataSet("hbias"));
    dataspace = dataset.getSpace();
    memspace = H5::DataSpace(1, fdim1);
    dataset.read(&_hbias[0], H5::PredType::NATIVE_FLOAT, memspace, dataspace);

    // Read visible unit biases
    dataset = H5::DataSet(h5file->openDataSet("vbias"));
    dataspace = dataset.getSpace();
    memspace = H5::DataSpace(2, fdim2);
    dataset.read(_vbias._data, H5::PredType::NATIVE_FLOAT, memspace, dataspace);

    // Read D
    fdim2[0] = _numVisible;
    fdim2[1] = _numHidden;
    dataset = H5::DataSet(h5file->openDataSet("D"));
    dataspace = dataset.getSpace();
    memspace = H5::DataSpace(2, fdim2);
    dataset.read(_D._data, H5::PredType::NATIVE_FLOAT, memspace, dataspace);

    // Clean HDF5
    delete h5file;
}

// end of file
