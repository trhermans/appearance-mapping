CC = gcc
CXXC = g++
CFLAGS = -Wall -g -O3

all: bsvm-train bsvm-predict svm-scale

# changing svm-predict.c to .cpp by Jianxin Wu, to cope with MS compiler.
bsvm-predict: svm-predict.cpp bsvm.a svm.h
	$(CXXC) $(CFLAGS) svm-predict.cpp bsvm.o solvebqp.o \
dtron/tron.a f2c/f2c.a -o bsvm-predict -lm
bsvm-train: svm-train.c bsvm.o bsvm.a svm.h 
#	Changed by Jianxin Wu to avoid g++ warning
#	$(CXXC) $(CFLAGS) svm-train.c bsvm.o solvebqp.o dtron/tron.a f2c/f2c.a -o bsvm-train -lm
	$(CXXC) $(CFLAGS) -U_FORTIFY_SOURCE svm-train.c bsvm.o solvebqp.o \
dtron/tron.a f2c/f2c.a -o bsvm-train -lm
svm-scale: svm-scale.c
#	Changed by Jianxin Wu to avoid g++ warning
#	$(CXXC) $(CFLAGS) svm-scale.c -o svm-scale
	$(CXXC) $(CFLAGS) -U_FORTIFY_SOURCE svm-scale.c -o svm-scale
bsvm.a: tron solvebqp.o bsvm.o
	
bsvm.o: bsvm.cpp svm.h
#	$(CXXC) $(CFLAGS) -c bsvm.cpp
#	Changed by Jianxin Wu to avoid g++ warning
	$(CXXC) $(CFLAGS) -U_FORTIFY_SOURCE -c bsvm.cpp
solvebqp.o: solvebqp.c svm.h
	$(CC) $(CFLAGS) -c solvebqp.c
tron: svm.h
	cd dtron; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';
	cd f2c; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';
clean:
	cd f2c;		make clean
	cd dtron;	make clean
	rm -f *~ bsvm.o solvebqp.o bsvm-train bsvm-predict svm-scale
