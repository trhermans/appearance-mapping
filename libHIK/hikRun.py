#!/usr/bin/env python
import sys
import os

LIN_MODEL_CMD = "./svm-train -c 0.03125 -t 0 "
POL_MODEL_CMD = "./svm-train -c 0.03125 -t 1 "
RBF_MODEL_CMD = "./svm-train -c 0.03125 -t 2 "
SIG_MODEL_CMD = "./svm-train -c 0.03125 -t 3 "

HIK_MODEL_CMD = "./svm-train -c 0.03125 -t 5 "

PREDICT_CMD = "./svm-predict "

def main(args):
    for i in range(1,6):
        cmd1 = RBF_MODEL_CMD + "train" + str(i) + ".txt.int"
        os.system(cmd1)
    for i in range(1,6):
        print "Results for test:", str(i)
        cmd2 = PREDICT_CMD + "test" + str(i) + ".txt.int train" + str(i) + ".txt.int.model out" + str(i) + ".txt"
        cmd3 = PREDICT_CMD + "train" + str(i) + ".txt.int train" + str(i) + ".txt.int.model out.train" + str(i) + ".txt"
        os.system(cmd2)
        os.system(cmd3)

if __name__ =='__main__':
    main(sys.argv[1:])
