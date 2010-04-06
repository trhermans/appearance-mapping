C++ = g++

C++-OPT = -O3 -DNDEBUG
C++-DEBUG = -g -O0
C++-WARNS = -Wall
C++-FLAGS = $(C++-WARNS) $(C++-OPT)
#C++-FLAGS = $(C++-WARNS) $(C++-DEBUG)
RM = rm -f
MKDIR = mkdir -p

PLATFORM =$(shell uname)
ifeq ($(PLATFORM), Darwin)
# OS X Includes
CV_INCLUDE = -I/usr/local/include -I/opt/local/include
else
CV_INCLUDE = -I/usr/local/include
endif
INCLUDE = -I./ $(CV_INCLUDE) $(VL_INCLUDE) -I./ -I./vis_codebook -I./fabmap

# Libraries
ifeq ($(PLATFORM), Darwin)
LIB_DIRS = -L./build -L/usr/local/lib -L/opt/local/lib
else
LIB_DIRS = -L./build -L/usr/local/lib
endif

CV_LIBS = -lcv -lcvaux -lcxcore -lhighgui
LDLIBS = $(LIB_DIRS) $(CV_LIBS)

# Directory to place all object files
OBJ_DIR = ./build

# Source file listings
GEN_CODEBOOK_SRCS = generate_codebook.cpp
TEST_CODEBOOK_SRCS = test_codebook_read.cpp
VIS_CODEBOOK_SRCS = vis_codebook/visual_codebook.cpp \
		    vis_codebook/visual_codebook.h
FABMAP_MAIN_SRCS = FabMapMain.cpp
FABMAP_CLASS_SRCS = fabmap/FABMAP.cpp \
		    fabmap/FABMAP.h
CHOW_LUI_SRCS = fabmap/ChowLiuTree.cpp \
		fabmap/ChowLiuTree.h
NAIVE_BAYES_SRCS = fabmap/NaiveBayes.cpp \
		   fabmap/NaiveBayes.h
DETECTOR_MODEL_SRCS = fabmap/DetectorModel.cpp \
		      fabmap/DetectorModel.h
PLACE_MODEL_SRCS = fabmap/PlaceModel.cpp \
		   fabmap/PlaceModel.h
SIMPLE_LOCATION_SRCS = fabmap/SimpleLocationPrior.cpp \
		       fabmap/SimpleLocationPrior.h
NORMALIZATION_SRCS = fabmap/NormalizationTermSampling.cpp \
		     fabmap/NormalizationTermSampling.h
# List of object files
OBJS =  $(OBJ_DIR)/visual_codebook.o \
	$(OBJ_DIR)/chow_lui.o \
	$(OBJ_DIR)/fabmap_class.o \
	$(OBJ_DIR)/naive_bayes.o \
	$(OBJ_DIR)/detector_model.o \
	$(OBJ_DIR)/place_model.o \
	$(OBJ_DIR)/simple_location_prior.o \
	$(OBJ_DIR)/normalization_term.o

# List of executables
EXECS = generate_codebook test_codebook_read fabmap_exec

# Compilation rules
default: $(EXECS)

#
# Codebook generation executable
#
generate_codebook: $(OBJ_DIR)/generate_codebook.o build_dir
	$(C++) $(C++-FLAGS) $(INCLUDE) $(LDLIBS) $(OBJ_DIR)/generate_codebook.o  $(OBJS) -o $@
test_codebook_read: $(OBJ_DIR)/test_codebook_read.o build_dir
	$(C++) $(C++-FLAGS) $(INCLUDE) $(LDLIBS) $(OBJ_DIR)/test_codebook_read.o  $(OBJS) -o $@
fabmap_exec: $(OBJ_DIR)/fabmap.o build_dir
	$(C++) $(C++-FLAGS) $(INCLUDE) $(LDLIBS) $(OBJ_DIR)/fabmap.o $(OBJS) -o $@
#
# EXEC objects
#
$(OBJ_DIR)/generate_codebook.o: $(GEN_CODEBOOK_SRCS) $(OBJS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/test_codebook_read.o: $(TEST_CODEBOOK_SRCS) $(OBJS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/fabmap.o: $(FABMAP_MAIN_SRCS) $(OBJS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

#
# Class Objects
#
$(OBJ_DIR)/fabmap_class.o: $(FABMAP_CLASS_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/chow_lui.o: $(CHOW_LUI_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/naive_bayes.o: $(NAIVE_BAYES_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/detector_model.o: $(DETECTOR_MODEL_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/place_model.o: $(PLACE_MODEL_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/simple_location_prior.o: $(SIMPLE_LOCATION_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/normalization_term.o: $(NORMALIZATION_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/visual_codebook.o: $(VIS_CODEBOOK_SRCS)
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

.Phony : clean build_dir

build_dir :
	$(MKDIR) $(OBJ_DIR)

clean :
	$(RM) $(EXECS) $(OBJ_DIR)/*.o