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
INCLUDE = -I./ $(CV_INCLUDE) $(VL_INCLUDE) -I./ -I./vis_codebook

# Libraries
ifeq ($(PLATFORM), Darwin)
LIB_DIRS = -L./build -L/usr/local/lib -L/opt/local/lib
else
LIB_DIRS = -L./build -L/usr/local/lib
endif

CV_LIBS = -lcv -lcvaux -lcxcore -lhighgui
VL_LIBS = -lvl
LDLIBS = $(LIB_DIRS) $(CV_LIBS) #$(VL_LIBS)

# Directory to place all object files
OBJ_DIR = ./build

# Source file listings
GEN_CODEBOOK_SRCS = generate_codebook.cpp
VIS_CODEBOOK_SRCS = vis_codebook/visual_codebook.cpp \
		    vis_codebook/visual_codebook.h

# List of object files
OBJS =  $(OBJ_DIR)/visual_codebook.o

# List of executables
EXECS = generate_codebook

# Compilation rules
default: $(EXECS)

#
# Codebook generation executable
#
generate_codebook: $(OBJ_DIR)/generate_codebook.o
	$(C++) $(C++-FLAGS) $(INCLUDE) $(LDLIBS) $(OBJ_DIR)/generate_codebook.o  $(OBJS) -o $@

$(OBJ_DIR)/generate_codebook.o: $(GEN_CODEBOOK_SRCS) $(OBJS) build_dir
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

$(OBJ_DIR)/visual_codebook.o: $(VIS_CODEBOOK_SRCS) build_dir
	$(C++) $(C++-FLAGS) $(INCLUDE) -c $< -o $@

.Phony : clean build_dir

build_dir :
	$(MKDIR) $(OBJ_DIR)

clean :
	$(RM) $(EXECS) $(OBJ_DIR)/*.o