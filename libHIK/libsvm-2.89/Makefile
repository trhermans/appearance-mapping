CXX? = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 1

all: svm-train svm-predict svm-scale

lib: svm.o
	$(CXX) -shared svm.o -o libsvm.so.$(SHVER)

svm-predict: svm-predict.cpp svm.o
# changing svm-predict.c to .cpp by Jianxin Wu, to cope with MS compiler.
	$(CXX) $(CFLAGS) svm-predict.cpp svm.o -o svm-predict -lm
svm-train: svm-train.cpp svm.o
# changing svm-train.c to .cpp by Jianxin Wu, to cope with MS compiler.
	$(CXX) $(CFLAGS) svm-train.cpp svm.o -o svm-train -lm
svm-scale: svm-scale.c
#	$(CXX) $(CFLAGS) svm-scale.c -o svm-scale
	$(CXX) $(CFLAGS) -U_FORTIFY_SOURCE svm-scale.c -o svm-scale
svm.o: svm.cpp svm.h
#	$(CXX) $(CFLAGS) -c svm.cpp
	$(CXX) $(CFLAGS) -U_FORTIFY_SOURCE -c svm.cpp
# The above two lines changed by Jianxin Wu, to suppress "ignoring return value" warning of gcc
clean:
	rm -f *~ svm.o svm-train svm-predict svm-scale
