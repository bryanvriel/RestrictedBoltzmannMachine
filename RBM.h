//-*- coding: C++ -*-

#ifndef RBM_h
#define RBM_h

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "Matrix.h"

struct ratings_t {
    int userId;
    int movieId;
    int rating;
};

struct user_t {
    int userId;
    std::vector<int> movieIds;
    std::vector<int> ratings;
};


class RBM {

    // Some useful typedefs
    public:
        typedef std::vector<float> vector_t;
        typedef std::vector<unsigned int> vector_int_t;

    // Define public variables (accessible by the user)
    public:
        // Constructor
        RBM(size_t numHidden, size_t numVisible, size_t K);
        // Allocator
        void allocate();
        // Bias initialization
        void initBiases(std::vector<user_t> & users);
        // Training
        void train(std::vector<user_t> & users, double lRate, unsigned short nCD, float wPenalty);
        // Predicting
        void predict(std::vector<user_t> &, std::vector<user_t> &, vector_t &);
        
        // Saving and loading states
        void saveState(char *) const;
        void loadState(const std::string);

        // Destructor
        ~RBM();

    // Define private variables
    protected:
        size_t _numHidden;
        size_t _numVisible;
        size_t _K;
        Matrix _W;
        Matrix _vbias;
        vector_t _hbias;
        std::vector<size_t> _movieRcount;
        const gsl_rng_type * _Trand;
        gsl_rng * _rng;

        // Compute one visible probability from hidden units
        void _sample_v_from_h(vector_t &, vector_t &, size_t);

        // Sigmoid logistic function 
        inline float _sigmoid(float arg) {
            return 1.0 / (1.0 + exp(-1.0 * arg));
        }

    // Define private functions
    private:
        // Updating
        float _update(std::vector<int> & ratings, std::vector<int> & movieIds, 
                      std::vector<float> & visible_units, vector_t & pos_hidden_probs, 
                      vector_t & neg_hidden_probs, float learningRate, float wPenalty);

    // Disable copy and assign operators
    private:
        RBM(const RBM &);
        const RBM & operator=(const RBM &);

};


class RBMcond : public RBM {
    /*
    Conditional RBM derived from the basic RBM. All we add is the D matrix which connects
    the visible units to the hidden units, no rating needed.
    */
    
    // Define public variables (accessible by the user)
    public:
        // Constructor
        RBMcond(size_t numHidden, size_t numVisible, size_t K);
        // Allocator
        void allocate();
        // Training
        void train(std::vector<user_t> & users, double lRate, unsigned short nCD, 
            float momentum, pthread_mutex_t * mutex);
        // Predicting
        void predict(std::vector<user_t> &, std::vector<user_t> &, vector_t &);
        // Destructor
        ~RBMcond() {_dHbias.clear();}

        // Saving and loading states
        void saveState(char *) const;
        void loadState(const std::string);

    // Define private variables
    protected:
        Matrix _D;          // matrix for modeling conditional effects
        Matrix _dW;         // Incremental weights (for momentum)
        Matrix _dVbias;
        vector_t _dHbias;
        Matrix _dD;

    // Define private functions
    private:
        // Updating
        float _update(std::vector<int> & ratings, std::vector<int> & movieIds, 
                      std::vector<float> & visible_units, vector_t & pos_hidden_probs, 
                      vector_t & neg_hidden_probs, double lRate_factor, float momentum);

    // Disable copy and assign operators
    private:
        RBMcond(const RBMcond &);
        const RBMcond & operator=(const RBMcond &);

};


class RBMcondFact : public RBM {
    /*
    Conditional RBM derived from the basic RBM. All we add is the D matrix which connects
    the visible units to the hidden units, no rating needed. This is also the factored 
    version to reduce the number of free parameters.
    */
    
    // Define public variables (accessible by the user)
    public:
        // Constructor
        RBMcondFact(size_t numHidden, size_t numVisible, size_t numFactor, size_t K);
        // Allocator
        void allocate();
        // Training
        void train(std::vector<user_t> & users, double lRate, unsigned short nCD);
        // Predicting
        void predict(std::vector<user_t> &, std::vector<user_t> &, vector_t &);
        // Destructor
        ~RBMcondFact() {};

        // Saving and loading states
        void saveState(char *) const;
        void loadState(const std::string);

    // Define private variables
    protected:
        size_t _numFactor;
        Matrix _A;
        Matrix _B;
        Matrix _D;      // matrix for modeling conditional effects

    // Define private functions
    private:
        // General updating
        float _update(std::vector<int> & ratings, std::vector<int> & movieIds, 
                      std::vector<float> & visible_units, vector_t & pos_hidden_probs, 
                      vector_t & neg_hidden_probs, double lRate_factor);
        // Update W given A and B
        void _updateW();

    // Disable copy and assign operators
    private:
        RBMcondFact(const RBMcondFact &);
        const RBMcondFact & operator=(const RBMcondFact &);

};

struct common {
    pthread_mutex_t mutex;
    float momentum;
    double lRate_factor;
    unsigned short nCD;    
    RBMcond * rbm;
    size_t nproc;
};


struct context {
    size_t istart;
    size_t iend;
    common * info;
    std::vector<user_t> users;
};


void thread_train(RBMcond * rbm, std::vector<user_t> & allUsers, double lRate_factor,
    unsigned short nCD, float momentum, size_t nproc);



#endif

// end of file
