#!/usr/bin/env python
import os
CMD_BASE = "./build_codebook "
PREFIX = "/home/thermans/src/fab-data/"
NC_IMAGES = PREFIX + "new-college/Images/"
NC_NUM_IMAGES = "2146"
CC_IMAGES = PREFIX + "city-centre/Images/"
CC_NUM_IMAGES = "2474"
USE_HARRIS = True

def main():
    #ks = [2000, 3000, 4000, 6000, 7000]
    descriptors = ["SIFT","CENTRSIT"]
    cluster = "kmeans" # kmeans kmedian HIK
    use_harris = "1" # 0 1
    k = 11000
    for descriptor in descriptors:
        train_file = PREFIX + "nc-" + descriptor + "-" + cluster + "-" + str(k)\
            + "-" +  use_harris + ".train"
        test_file = PREFIX + "cc-" + descriptor + "-" + cluster + "-" + str(k)\
            + "-" + use_harris + ".test"
        cmd = CMD_BASE + NC_IMAGES + " " + NC_NUM_IMAGES + " " + CC_IMAGES \
            + " " + CC_NUM_IMAGES + " " + str(k) + " " + descriptor +" " \
            + cluster + " " + train_file + " " + test_file + " "  + use_harris

        print cmd
        os.system(cmd)

if __name__ == '__main__':
    main()
