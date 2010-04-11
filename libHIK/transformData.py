#!/usr/bin/env python

import sys
import os
CMD = "./transform "
#CMD = "echo "
def main(args):
    for i in range(1,6):
        ech = CMD + "train" + str(i) + ".txt test" + str(i) + ".txt"
        os.system(ech)

if __name__ =='__main__':
    main(sys.argv[1:])

