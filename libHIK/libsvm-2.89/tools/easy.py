#!/usr/bin/env python

import sys
import os
from subprocess import *

if len(sys.argv) <= 1:
	print 'Usage: %s training_file [testing_file]' % sys.argv[0]
	raise SystemExit

# svm, grid, and gnuplot executable files

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmscale_exe = "../svm-scale"
	svmtrain_exe = "../svm-train"
	svmpredict_exe = "../svm-predict"
	grid_py = "./grid.py"
	gnuplot_exe = "/usr/bin/gnuplot"
else:
        # example for windows
	svmscale_exe = r"..\windows\svm-scale.exe"
	svmtrain_exe = r"..\windows\svm-train.exe"
	svmpredict_exe = r"..\windows\svm-predict.exe"
	gnuplot_exe = r"c:\tmp\gnuplot\bin\pgnuplot.exe"
	grid_py = r".\grid.py"

assert os.path.exists(svmscale_exe),"svm-scale executable not found"
assert os.path.exists(svmtrain_exe),"svm-train executable not found"
assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
assert os.path.exists(gnuplot_exe),"gnuplot executable not found"
assert os.path.exists(grid_py),"grid.py not found"

train_pathname = sys.argv[1]
assert os.path.exists(train_pathname),"training file not found"
file_name = os.path.split(train_pathname)[1]
scaled_file = file_name + ".scale"
model_file = file_name + ".model"
range_file = file_name + ".range"

if len(sys.argv) > 2:
	test_pathname = sys.argv[2]
	file_name = os.path.split(test_pathname)[1]
	assert os.path.exists(test_pathname),"testing file not found"
	scaled_test_file = file_name + ".scale"
	predict_test_file = file_name + ".predict"

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

# Changed to a smaller (C,g) range for faster processing by Jianxin Wu    
cmd = '%s -svmtrain "%s" -log2c 1,11,2 -log2g -5,-15,-2 -gnuplot "%s" "%s"' % (grid_py, svmtrain_exe, gnuplot_exe, scaled_file)
print 'Cross validation...'
f = Popen(cmd, shell = True, stdout = PIPE).stdout

line = ''
while True:
	last_line = line
	line = f.readline()
	if not line: break
c,g,rate = map(float,last_line.split())

print 'Best c=%s, g=%s CV rate=%s' % (c,g,rate)

cmd = '%s -c %s -g %s "%s" "%s"' % (svmtrain_exe,c,g,scaled_file,model_file)
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

	cmd = '%s "%s" "%s" "%s"' % (svmpredict_exe, scaled_test_file, model_file, predict_test_file)
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
