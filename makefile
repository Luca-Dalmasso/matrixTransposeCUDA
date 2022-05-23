
#Author: Dalmasso Luca
#Last modification: 16/05/2022 by Luca

#Application name (main.cu file by default)
NAME=matrixTranspose
#Source path (.cu)
SRC=src/
#Headers path (.h)
INCLUDES=inc/
#Build path (.o)
BUILD=build/
#Software documentation path (html and LaTex format only)
DOCS=docs/
#CUDA compiler path
NVCC=/usr/local/cuda/bin/nvcc


#COMPILE FLAGS
CUDA_FLAGS=
#change DEBUG flag to 0 to disable debug simbols (DO NOT ENABLE DEBUG SIMBOLS WHEN PROFILING)
DEBUG=0

ifeq ($(DEBUG),1)
	CUDA_FLAGS+=-g
	CUDA_FLAGS+=-G
	CUDA_FLAGS+=-DDEBUG
else
	CUDA_FLAGS+=-O2	
endif

#change L1EN flag to 0 to disable L1 cache
L1EN=1

ifeq ($(L1EN),1)
	CUDA_FLAGS+=-Xptxas -dlcm=ca
else
	CUDA_FLAGS+=-Xptxas -dlcm=cg
endif

#AVAILABLE COMMANDS:
#all: compilation process (DEFAULT)
#clean: cleaning process will clean build/ (TO BE EXPLICITLY CALLED es: make clean)
#docs: build documantation in html and LaTex, documentation will be saved in docs/, (TO BE EXPLICITLY CALLED es: make docs)
#test: just used to check that all paths are correct (TO BE EXPLICITLY CALLED es: make test)
.PHONY: all clean docs test

#list of all CUDA source files
SOURCES=$(wildcard $(SRC)*.cu)
#redirect the sources path to build/
BUILD_PATH=$(subst $(SRC),$(BUILD),$(SOURCES))
#path where object files are going to be placed
OBJECTS=$(patsubst %.cu, %.o, $(BUILD_PATH))
#list of all header files
HEADERS=$(wildcard $(INCLUDES)*.h)
	
#default commands (compilation)
all: $(BUILD)$(NAME).o $(OBJECTS)
	@echo $@;
	$(NVCC) $^ -o $(NAME)

$(BUILD)$(NAME).o: $(NAME).cu
	@echo $@;
	$(NVCC) $(CUDA_FLAGS) -I$(INCLUDES) -c $< -o $@

$(BUILD)%.o: $(SRC)%.cu $(HEADERS)
	@echo $@;
	$(NVCC) $(CUDA_FLAGS) -I$(INCLUDES) -c $< -o $@

#docs command
docs: Doxifile $(HEADERS) $(SOURCES)
	@echo $@;
	doxygen $<
	
#test command
test:
	@echo "SOURCES: $(SOURCES)";
	@echo "OBJECTS: $(OBJECTS)";
	@echo "HEADERS: $(HEADERS)";
	@echo "CUDA FLAGS: $(CUDA_FLAGS)";

#clean command
clean:
	@echo $@;
	rm -rf $(BUILD)*
	rm -f $(NAME).o
	rm -f $(NAME)

	
