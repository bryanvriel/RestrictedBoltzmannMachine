// -*- C++ -*-

#ifndef MATRIX_H
#define MATRIX_H

// Simple Matrix class implementation

#include <cstdlib>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

class Matrix {

    public:
        // Constructors
        inline Matrix() {};
        Matrix(size_t rows, size_t cols);
        // Destructor
        ~Matrix();

        // Subscript operators
        inline float& operator() (size_t row, size_t col) {
            return _data[row * _cols + col];
        }
        inline float operator() (size_t row, size_t col) const {
            return _data[row * _cols + col];
        }

        // Allocation
        void allocate(size_t rows, size_t cols);
        // Fill with values
        void zeros();
        void random(float mean, double std);

        // Transpose
        void transpose(float *);

        // Matrix multiply given pointers to two other Matrices
        void multiply(Matrix * A, Matrix * B);

        // Getting dimensions
        inline size_t rows() {return _rows;}
        inline size_t cols() {return _cols;}

        // Make data visible, but put underscore on it to discourage access :)
        float * _data;

    private:
        size_t _rows, _cols;

    // Disable copy and assign operators
    private:
        Matrix(const Matrix &);
        const Matrix & operator=(const Matrix &);

};

#endif

// end of file
