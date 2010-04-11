#!/usr/bin/env python

# This file is a tool revised from the easy.py script by Jianxin Wu
# It is mainly for my personal use, but you can use it if you like
# Usage: python train_test.py training_file testing_file
# Because I am lazy, you need to change this script to set the parameters, instead of passing them as options

import sys
import os
from subprocess import *

if len(sys.argv) <= 2:
	print 'Usage: %s training_file testing_file' % sys.argv[0]
	raise SystemExit

# svm executable files

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmscale_exe = "../svm-scale"
	svmtrain_exe = "../svm-train"
	svmpredict_exe = "../svm-predict"
else:
        # example for windows
	svmscale_exe = r"..\windows\svm-scale.exe"
	svmtrain_exe = r"..\windows\svm-train.exe"
	svmpredict_exe = r"..\windows\svm-predict.exe"

assert os.path.exists(svmscale_exe),"svm-scale executable not found"
assert os.path.exists(svmtrain_exe),"svm-train executable not found"
assert os.path.exists(svmpredict_exe),"svm-predict executable not found"

train_pathname = sys.argv[1]
assert os.path.exists(train_pathname),"training file not found"
file_name = os.path.split(train_pathname)[1]
scaled_file = file_name + ".scale"
model_file = file_name + ".model"
range_file = file_name + ".range"

test_pathname = sys.argv[2]
file_name = os.path.split(test_pathname)[1]
assert os.path.exists(test_pathname),"testing file not found"
scaled_test_file = file_name + ".scale"
predict_test_file = file_name + ".predict"

c = 0.03125
g = 0.0078125
t = 5
b = 1
# set scale = 0 (default) if do not want to scale the data (or you have scaled it)
# set scale = 1 if otherwise, added by Jianxin Wu
scale = 0

if scale > 0:
    cmd = '%s -s "%s" "%s" > "%s"' % (svmscale_exe, range_file, train_pathname, scaled_file)
    print 'Scaling training data...'
    Popen(cmd, shell = True, stdout = PIPE).communicate()	
else:
    # this "else" branch is added by Jianxin Wu
    scaled_file = train_pathname
    
cmd = '%s -b %s -t %s -c %s -g %s "%s" "%s"' % (svmtrain_exe,b,t,c,g,scaled_file,model_file)
print 'Training...'
Popen(cmd, shell = True, stdout = PIPE).communicate()

print 'Output model: %s' % model_file
if len(sys.argv) > 2:
    if scale > 0:
        cmd = '%s -r "%s" "%s" > "%s"' % (svmscale_exe, range_file, test_pathname, scaled_test_file)
        print 'Scaling testing data...'
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
    else:
        scaled_test_file = test_pathname

	cmd = '%s -b %s "%s" "%s" "%s"' % (svmpredict_exe, b, scaled_test_file, model_file, predict_test_file)
	print 'Testing...'
	Popen(cmd, shell = True).communicate()	

	print 'Output prediction: %s' % predict_test_file

# Below added by Jianxin Wu
# sometimes it is useful to remove these generated files automatically

if not is_win32:
	del_exe = 'rm -f'
else:
    del_exe = 'del'

cmd = "%s %s" % (del_exe, range_file)
Popen(cmd, shell = True).communicate()	

if scale > 0:
    cmd = "%s %s" % (del_exe, scaled_file)
    Popen(cmd, shell = True).communicate()	

cmd = "%s %s" % (del_exe, model_file)
Popen(cmd, shell = True).communicate()	

if scale > 0:
    cmd = "%s %s" % (del_exe, scaled_test_file)
    Popen(cmd, shell = True).communicate()	

cmd = "%s %s" % (del_exe, predict_test_file)
Popen(cmd, shell = True).communicate()	

cmd = "rm -f *.out"
os.system(cmd)

cmd = "rm -f *.png"
os.system(cmd)
