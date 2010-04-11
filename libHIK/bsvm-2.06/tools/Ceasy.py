#!/usr/bin/env python

import sys
import os

if len(sys.argv) <= 1:
	print 'Usage: %s training_file [testing_file]' % sys.argv[0]
	raise SystemExit

# svm, grid, and gnuplot executable

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmscale_exe = "../svm-scale"
	svmtrain_exe = "../bsvm-train"
	svmpredict_exe = "../bsvm-predict"
	grid_py = "./grid.py"
	gnuplot_exe = "/usr/bin/gnuplot"
else:
        # example for windows
	svmscale_exe = r"..\windows\svmscale.exe"
	svmtrain_exe = r"..\windows\svmtrain.exe"
	svmpredict_exe = r"..\windows\svmpredict.exe"
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

scale = 0

if scale > 0:
	cmd = "%s -s %s %s > %s" % (svmscale_exe, range_file, train_pathname, scaled_file)
	print 'Scaling training data...'
	os.system(cmd)
else:
	cmd = "cp -f %s %s" % (train_pathname,scaled_file)
	os.system(cmd)

svmtype = 2
t = 5

cmd = "%s -s %s -log2c -5,11,2 -log2g 1,1,1 -t %s -gnuplot %s %s" % (grid_py, svmtype, t, gnuplot_exe, scaled_file)
print 'Cross validation...'
dummy, f = os.popen2(cmd)

line = ''
while 1:
	last_line = line
	line = f.readline()
	if not line: break
c,g,rate = map(float,last_line.split())

print 'Best c=%s, g=%s CV rate=%s' % (c,g,rate)

cmd = "%s -s %s -c %s -g %s -t %s %s %s" % (svmtrain_exe,svmtype,c,g,t,scaled_file,model_file)
print 'Training...'
os.popen(cmd)

print 'Output model: %s' % model_file
if len(sys.argv) > 2:
	if scale > 0:
		cmd = "%s -r %s %s > %s" % (svmscale_exe, range_file, test_pathname, scaled_test_file)
		print 'Scaling testing data...'
		os.system(cmd)
	else:
		cmd = "cp -f %s %s" % (test_pathname,scaled_test_file)
		os.system(cmd)

	cmd = "%s %s %s %s" % (svmpredict_exe, scaled_test_file, model_file, predict_test_file)
	print 'Testing...'
	os.system(cmd)

	print 'Output prediction: %s' % predict_test_file
	
cmd = "rm -f %s" % (range_file)
os.system(cmd)

cmd = "rm -f %s" % (scaled_file)
os.system(cmd)

cmd = "rm -f %s" % (model_file)
os.system(cmd)

cmd = "rm -f %s" % (scaled_test_file)
os.system(cmd)

cmd = "rm -f %s" % (predict_test_file)
os.system(cmd)

cmd = "rm -f *.out"
os.system(cmd)

cmd = "rm -f *.png"
os.system(cmd)
