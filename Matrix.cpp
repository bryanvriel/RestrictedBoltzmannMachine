// -*- C++ -*-

#include <gsl/gsl_cblas.h>
#include "Matrix.h"

Matrix::Matrix(size_t rows, size_t cols) : _rows(rows), _cols(cols) {
    /*
    Simple constructor allocates the data.
    */
    _data = new float[rows*cols];
}


void Matrix::allocate(size_t rows, size_t cols) {
    /*
    After construction, allocate data
    */
    _rows = rows;
    _cols = cols;
    _data = new float[rows*cols];
}


void Matrix::zeros() {
    /*
    Fill values with zeros.
    */
    for (size_t i = 0; i < (_rows * _cols); ++i) {
        _data[i] = 0.0;
    }
}


void Matrix::random(float mean, double std) {
    /*
    Fill matrix with Gaussian random values with specified mean and standard deviation.
    */
    const gsl_rng_type * T = gsl_rng_mt19937;
    gsl_rng * rng = gsl_rng_alloc(T);
    for (size_t i = 0; i < (_rows * _cols); ++i) {
        _data[i] = (float) gsl_ran_gaussian(rng, std) + mean;
    }
    gsl_rng_free(rng);
}


void Matrix::transpose(float * dest) {
    /*
    Transpose data and store in another matrix data.
    */
    for (size_t n = 0; n < _rows * _cols; ++n) {
        size_t i = n / _rows;
        size_t j = n % _rows;
        dest[n] = _data[_cols*j + i];
    }
}


void Matrix::multiply(Matrix * A, Matrix * B) {
    /*
    Multiply two other matrices together and save in self.
    */

    // Get matrix sizes
    size_t Arow = A->rows();
    size_t Acol = A->cols();
    size_t Bcol = B->cols();

    // Call CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Arow, Bcol, Acol, 1.0, A->_data,
        Acol, B->_data, Bcol, 0.0, _data, _cols);

}


Matrix::~Matrix() {
    /*
    Destructor: delete the dynamic memory.
    */
    delete [] _data;
}

// end of file
