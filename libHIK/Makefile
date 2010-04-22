CC = g++
WLECEL = -Wall
CFLAGS = -c $(WLEVEL) -O3
#CFLAGS = -c $(WLEVEL) -g3
SOURCES = hik_svm.cpp hik-predict.cpp UseSVM.cpp transform.cpp\
 svm.cpp hik-bsvm-predict.cpp transform.cpp \
 HistogramKernel.cpp LinearKernel.cpp Cluster.cpp util.cpp \
 hik-cluster.cpp Features.cpp Codebook.cpp Matrix.cpp \
 Datasets.cpp Generate_Data.cpp build_codebook.cpp

all: transform hik-predict hik-bsvm-predict hik-cluster generate-data SVMs build_codebook

# depend:
# 	g++ $(CFLAGS) -MM $(SOURCES) > .deps

transform: transform.cpp
	$(CC) $(WLEVEL) -O3 transform.cpp -o transform

hik_svm.o:
	$(CC) $(CFLAGS) hik_svm.cpp -o hik_svm.o

UseSVM.o:
	$(CC) $(CFLAGS) UseSVM.cpp -o UseSVM.o

hik-predict.o:
	$(CC) $(CFLAGS) hik-predict.cpp -o hik-predict.o

svm.o:
	$(CC) $(CFLAGS) -U_FORTIFY_SOURCE svm.cpp -o svm.o

hik-predict: hik-predict.o hik_svm.o svm.o UseSVM.o
	$(CC) $(LDFLAGS) hik-predict.o hik_svm.o svm.o UseSVM.o -o hik-predict

hik-bsvm-precit.o:
	$(CC) $(CFLAGS) hik-bsvm-predict.cpp -o hik-bsvm-predict.o

hik-bsvm-predict: hik-bsvm-predict.o hik_svm.o svm.o UseSVM.o
	$(CC) $(LDFLAGS) hik-bsvm-predict.o hik_svm.o svm.o UseSVM.o -o hik-bsvm-predict

HistogramKernel.o: HistogramKernel.cpp
	$(CC) $(CFLAGS) -fopenmp HistogramKernel.cpp -o HistogramKernel.o

LinearKernel.o: LinearKernel.cpp
	$(CC) $(CFLAGS) -fopenmp LinearKernel.cpp -o LinearKernel.o

Cluster.o: Cluster.cpp
	$(CC) $(CFLAGS) Cluster.cpp -o Cluster.o

util.o: util.cpp
	$(CC) $(CFLAGS) util.cpp -o util.o

hik-cluster.o: hik-cluster.cpp
	$(CC) $(CFLAGS) hik-cluster.cpp -o hik-cluster.o

hik-cluster: hik-cluster.o HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Cluster.o
	$(CC) -fopenmp hik-cluster.o HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Cluster.o -o hik-cluster

Features.o: Features.cpp
	$(CC) $(CFLAGS) `pkg-config --cflags opencv` Features.cpp -o Features.o

Codebook.o: Codebook.cpp
	$(CC) $(CFLAGS) `pkg-config --cflags opencv` Codebook.cpp -o Codebook.o

Matrix.o: Matrix.cpp
	$(CC) $(CFLAGS) Matrix.cpp -o Matrix.o

Datasets.o: Datasets.cpp
	$(CC) $(CFLAGS) -fopenmp `pkg-config --cflags opencv` Datasets.cpp -o Datasets.o

Generate_Data.o: Generate_Data.cpp
	$(CC) $(CFLAGS) `pkg-config --cflags opencv` Generate_Data.cpp -o Generate_Data.o

generate-data: HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Matrix.o Features.o Codebook.o Datasets.o Generate_Data.o
	$(CC) -fopenmp `pkg-config --libs opencv` HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Matrix.o Features.o Codebook.o Datasets.o Generate_Data.o -o generate-data

build_codebook.o: build_codebook.cpp
	$(CC) $(CFLAGS) `pkg-config --cflags opencv` build_codebook.cpp -o build_codebook.o

build_codebook: HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Matrix.o Features.o Codebook.o build_codebook.o
	$(CC) -fopenmp `pkg-config --libs opencv` HistogramKernel.o LinearKernel.o svm.o util.o UseSVM.o Matrix.o Features.o Codebook.o build_codebook.o -o build_codebook

SVMs:
	cd libsvm-2.89; make; cp svm-train ../; cp svm-predict ../;
	cd bsvm-2.06; make; cp bsvm-train ../; cp bsvm-predict ../;

clean:
	rm -f *.o transform *~ *.bak hik-cluster hik-predict hik-bsvm-predict generate-data svm-train svm-predict bsvm-train bsvm-predict build_codebook
	cd libsvm-2.89; make clean;
	cd bsvm-2.06; make clean


#include .deps
