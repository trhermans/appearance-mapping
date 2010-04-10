#!/usr/bin/env python
import sys
import os
CMD1 = "./generate-data jaffe SIFT HIK"
CMD2 = "./transformData.py"
CMD3 = "./hikRun.py"

def main():
    os.system(CMD1)
    os.system(CMD2)
    os.system(CMD3)

if __name__ == '__main__':
    main()

