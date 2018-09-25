
CC=g++

CFLAGS=-I/opt/local/include -O2 -std=c++11 -Wall
LFLAGS=-L/opt/local/lib -lhdf5_cpp -lhdf5 -lgsl -lopenblas -lm

all: train train_cond predict predict_cond
.PHONY: all

train: train.cpp RBM.cpp Matrix.cpp
	${CC} ${CFLAGS} ${LFLAGS} train.cpp RBM.cpp Matrix.cpp -o train

train_cond: train_cond.cpp RBMcond.cpp RBM.cpp Matrix.cpp
	${CC} ${CFLAGS} ${LFLAGS} train_cond.cpp RBMcond.cpp RBM.cpp Matrix.cpp -o train_cond

predict: predict.cpp RBM.cpp Matrix.cpp
	${CC} ${CFLAGS} ${LFLAGS} predict.cpp RBM.cpp Matrix.cpp -o predict

predict_cond: predict_cond.cpp RBMcond.cpp RBM.cpp Matrix.cpp
	${CC} ${CFLAGS} ${LFLAGS} predict_cond.cpp RBMcond.cpp RBM.cpp Matrix.cpp -o predict_cond

clean:
	-@rm *.o train predict train_cond predict_cond 2> /dev/null || true
