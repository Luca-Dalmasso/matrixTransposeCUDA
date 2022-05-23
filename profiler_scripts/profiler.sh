#!/bin/bash

#profiler script for CUDA-C application
#1)generate a report:
#	| Kernel name | Elapsed Time (s)  | Bandwidth (MB/s) |
#	| ----------- | :--------------- :| :--------------: |
#2)generate report:
#	| Kernel name | Global Memory Profile | Shared Memory Profile | Occupancy | Divergence |
#   | ----------- | :-------------------: | :-------------------: | :--------:| :--------: |
#3)find best kernel, and on it perform grid level optimization and generate report
#	| Kernel name | Elapsed Time (s) | Bandwidth (MB/s) | Block(x,y) |
#   | ----------- | :--------------: | :--------------: | :--------: |


######################################################################
##
## SETUP
##
######################################################################
APP_NAME=matrixTranspose
CURRENT_DIR=.
REPORT_DIR=../${CURRENT_DIR}/report
LOG_DIR=../${CURRENT_DIR}/logs
ENABLE_GRID_OPT=0
PROFILER_PATH=/usr/local/cuda/bin/nvprof
BIN=../${CURRENT_DIR}/${APP_NAME}
REPORT_ID=$(date +%m-%d-%y_%H-%M-%S)
TMP_DIR=../${CURRENT_DIR}/tmp
#reports are saved as tablexx.rep
REPORT_FILE1=${REPORT_DIR}/${REPORT_ID}_t1.rep
REPORT_FILE2=${REPORT_DIR}/${REPORT_ID}_t2.rep
REPORT_FILE3=${REPORT_DIR}/${REPORT_ID}_t3.rep
REPORT_FILE4=${REPORT_DIR}/${REPORT_ID}_ot3.rep
#program errors log file
STDERR_LOG=${LOG_DIR}/${REPORT_ID}_err.log

source ${CURRENT_DIR}/setup.sh

######################################################################
##
## 1: run all kernels with default grid to collect first general datas
##
######################################################################

source ${CURRENT_DIR}/defgrid.sh

######################################################################
##
## 2: run all kernels with default grid and collect datas from nvprof
##
######################################################################

source ${CURRENT_DIR}/profgrid.sh

######################################################################
##
## 3: select best performant kernel from step1 and run grid level opt. 
##
######################################################################
source ${CURRENT_DIR}/findgrid.sh
