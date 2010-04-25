#!/usr/bin/env python
import os
CMD_BASE = "./build_codebook "
PREFIX = "/home/thermans/src/fab-data/"
NC_IMAGES = PREFIX + "new-college/Images/"
NC_NUM_IMAGES = "2146"
CC_IMAGES = PREFIX + "city-centre/Images/"
CC_NUM_IMAGES = "2474"

def main():
    #ks = [2000, 3000, 4000, 6000, 7000]
    #descriptors = ["SIFT","CENTRIST"]
    #descriptors = ["CENTRIST","SIFT"]
    descriptors = ["SIFT"]
    cluster = "kmeans" # kmeans kmedian HIK
    use_harris = "2" # 0 1 2
    k = 11000
    #k = 2
    for descriptor in descriptors:
        train_file1 = PREFIX + "nc-" + descriptor + "-" + cluster + "-" + str(k)\
            + "-" +  use_harris + ".train"
        test_file1 = PREFIX + "cc-" + descriptor + "-" + cluster + "-" + str(k)\
            + "-" + use_harris + ".test"
        cmd1 = CMD_BASE + NC_IMAGES + " " + NC_NUM_IMAGES + " " + CC_IMAGES \
            + " " + CC_NUM_IMAGES + " " + str(k) + " " + descriptor +" " \
            + cluster + " " + train_file1 + " " + test_file1 + " "  + use_harris

        print cmd1
        os.system(cmd1)

        # train_file2 = PREFIX + "cc-" + descriptor + "-" + cluster + "-" + str(k)\
        #     + "-" +  use_harris + ".train"
        # test_file2 = PREFIX + "nc-" + descriptor + "-" + cluster + "-" + str(k)\
        #     + "-" + use_harris + ".test"
        # cmd2 = CMD_BASE + CC_IMAGES + " " + CC_NUM_IMAGES + " " + NC_IMAGES \
        #     + " " + NC_NUM_IMAGES + " " + str(k) + " " + descriptor +" " \
        #     + cluster + " " + train_file2 + " " + test_file2 + " "  + use_harris

        # print cmd2
        # os.system(cmd2)

if __name__ == '__main__':
    main()
