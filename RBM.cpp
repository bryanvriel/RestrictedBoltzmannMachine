// -*- C++ -*-

#include <functional>
#include <algorithm>
#include "H5Cpp.h"
#include "RBM.h"


RBM::RBM(size_t nhidden, size_t nvisible, size_t korder) : 
    _numHidden(nhidden), _numVisible(nvisible), _K(korder) {
    /*
    Constructor
    */
}


void RBM::allocate() {
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

    _movieRcount.resize(_numVisible*_K, 0);

}


void RBM::initBiases(std::vector<user_t> & users) {
    /*
    Initialize the visible biases to log of average rating.
    */

    // Loop over the users
    size_t nusers = users.size();
    for (size_t uIndex = 0; uIndex < nusers; ++uIndex) {

        // Get the current user
        user_t user = users[uIndex];
        size_t nRatings = user.ratings.size();

        // Loop over the user's ratings and increment the corresponding movie counts
        for (size_t j = 0; j < nRatings; ++j) {
            int rating = user.ratings[j];
            if (rating > 0) {
                size_t offset = user.movieIds[j] * _K;
                _movieRcount[offset+rating-1] += 1;
            }
        }
    }

    // Loop over the movies
    for (size_t i = 0; i < _numVisible; ++i) {
        size_t mtotal = 0;
        for (size_t k = 0; k < _K; ++k) {
            mtotal += _movieRcount[i*_K + k];
        }
        for (size_t k = 0; k < _K; ++k) {
            float rcount = (float) _movieRcount[i*_K + k] + 1; // add 1 for numerical stability
            _vbias(i,k) = log(rcount / ((float) mtotal));
        }
    }


}

void RBM::train(std::vector<user_t> & users, double lRate, unsigned short nCD, float wPenalty) {
    /*
    Train the RBM given a list of user structurs.
    */
    size_t nusers = users.size();

    // Initialize some working vectors and variables
    vector_t hidden_probs(_numHidden,0.0);
    vector_t pos_hidden_probs(_numHidden,0.0);
    std::vector<double> temp_visible_prob(_K,0.0);
    vector_t visible_units;

    printf(" - training for %zu nusers\n", nusers);

    // Loop over the users
    float sumSquareError = 0.0;
    size_t totalRatings = 0;
    for (size_t uIndex = 0; uIndex < nusers; ++uIndex) {

        // Get the current user
        user_t user = users[uIndex];
        size_t nRatings = user.ratings.size();
        visible_units.resize(nRatings * _K);

        // Seed the Gibbs sampling with the training set to compute the hidden probabilities
        for (size_t j = 0; j < _numHidden; ++j) {
            float vWsum = 0.0;
            for (size_t i = 0; i < nRatings; ++i) {
                size_t rating0 = user.ratings[i] - 1;
                size_t offset = user.movieIds[i] * _K;
                vWsum += _W(offset+rating0,j);
            }
            hidden_probs[j]= _sigmoid(_hbias[j] + vWsum);
        }
        pos_hidden_probs = hidden_probs;

        // Now do rest of CD steps
        for (unsigned short T = 0; T < nCD; ++T) {

            // Sample the visible units given the hidden units
            for (size_t i = 0; i < nRatings; ++i) {
                size_t visibleIndex = user.movieIds[i];
                size_t offset = visibleIndex * _K;
                double sumProbability = 0.0;
                for (size_t k = 0; k < _K; ++k ) {
                    float hWsum = 0.0;
                    for (size_t j = 0; j < _numHidden; ++j) {
                        hWsum += hidden_probs[j] * _W(offset+k,j);
                    }
                    double expActivation = exp(_vbias(visibleIndex,k) + hWsum);
                    temp_visible_prob[k] = expActivation;
                    sumProbability += expActivation;
                }

                // Normalize the probabilities and copy to main visible unit
                for (size_t k = 0; k < _K; ++k) {
                    temp_visible_prob[k] /= sumProbability;
                }
                visible_units[i*_K  ] = temp_visible_prob[0];
                visible_units[i*_K+1] = temp_visible_prob[1];
                visible_units[i*_K+2] = temp_visible_prob[2];
                visible_units[i*_K+3] = temp_visible_prob[3];
                visible_units[i*_K+4] = temp_visible_prob[4];

            }

            // Sample the hidden probs and units given the visible units
            for (size_t j = 0; j < _numHidden; ++j) {
                float vWsum = 0.0;
                for (size_t i = 0; i < nRatings; ++i) {
                    size_t offset = user.movieIds[i] * _K;
                    vWsum += visible_units[i*_K    ] * _W(offset  ,j)
                           + visible_units[i*_K + 1] * _W(offset+1,j)
                           + visible_units[i*_K + 2] * _W(offset+2,j)
                           + visible_units[i*_K + 3] * _W(offset+3,j)
                           + visible_units[i*_K + 4] * _W(offset+4,j);
                }
                hidden_probs[j] = _sigmoid(_hbias[j] + vWsum);
            }

        }

        // Update the weights and biases
        float userError = _update(user.ratings, user.movieIds, visible_units, pos_hidden_probs, 
            hidden_probs, lRate, wPenalty);

        // Accumulate errors and count
        sumSquareError += userError;
        totalRatings += nRatings;
        
    }

     // Print the current error
    printf(" - training RMSE: %f\n", sqrt(sumSquareError / float(totalRatings)));

    // Clear the vectors
    hidden_probs.clear();
    pos_hidden_probs.clear();
    temp_visible_prob.clear();
    visible_units.clear();
    
}


void RBM::_sample_v_from_h(vector_t & hiddenUnit, vector_t & visibleProb, 
    size_t visibleIndex) {
    /*
    Compute visible unit PROBABILITIES only if output vector is floating point.
    */
    float sumProb = 0.0;
    float sum;
    size_t offset = visibleIndex * _K;
    for (size_t k = 0; k < _K; ++k) {
        sum = 0.0;
        for (size_t j = 0; j < _numHidden; ++j) {
            sum += hiddenUnit[j] * _W(offset+k,j);
        }
        float expActivation = exp(_vbias(visibleIndex,k) + sum);
        visibleProb[k] = expActivation;
        sumProb += expActivation;
    }

    // Normalize the probabilities
    for (size_t k = 0; k < _K; ++k) {
        visibleProb[k] /= sumProb;
    }

}


float RBM::_update(std::vector<int> & ratings, std::vector<int> & movieIds, 
    std::vector<float> & visible_units, vector_t & pos_hidden_probs, 
    vector_t & neg_hidden_probs, float learningRate, float wPenalty) {
    /*
    Update the weight matrix and biases.
    */
    size_t numRatings = ratings.size();
    float error = 0.0;
    float v0, vk;
    for (size_t i = 0; i < numRatings; ++i) {
        size_t visibleIndex = movieIds[i];
        size_t offset = visibleIndex * _K;
        int r = ratings[i] - 1;
        for (size_t k = 0; k < _K; ++k) {

            // Unpack the observed and predicted ratings
            if (int(k) == r) {
                v0 = 1.0;
            }
            else {
                v0 = 0.0;
            }
            vk = visible_units[i*_K + k];
           
            // Update the weights 
            for (size_t j = 0; j < _numHidden; ++j) {
                // Compute gradient
                float gradient = pos_hidden_probs[j] * v0 - neg_hidden_probs[j] * vk;
                float Wold = _W(offset+k,j);
                _W(offset+k,j) = Wold + learningRate * (gradient - wPenalty * Wold);
            }

            // Update the visible bias
            float deltaV = v0 - vk;
            _vbias(visibleIndex,k) += learningRate * deltaV;

            //// Accumulate the error
            error += deltaV * deltaV;

        }
    }

    // Update hidden unit biases separately
    for (size_t j = 0; j < _numHidden; ++j) {
        //float dhb = pos_hidden_probs[j] - neg_hidden_probs[j];
        //dhb = momentum * dhb + learningRate * dhb;
        float dhb = learningRate * (pos_hidden_probs[j] - neg_hidden_probs[j]);
        _hbias[j] += dhb;
    } 

    return error;
}


void RBM::predict(std::vector<user_t> & training_users, std::vector<user_t> & testing_users,
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
                int rating = user_train.ratings[i];
                size_t offset = user_train.movieIds[i] * _K;
                hsum += _W(offset + rating - 1, j);
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




void RBM::saveState(char * filename) const {
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

    // Clean HDF5
    delete h5file;

}


void RBM::loadState(const std::string filename) {
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

    // Clean HDF5
    delete h5file;
}


RBM::~RBM() {
    /*
    Destructor - clear the vectors for the weight matrix and biases.
    */
    _hbias.clear();
    _movieRcount.clear();
    gsl_rng_free(_rng);
}
