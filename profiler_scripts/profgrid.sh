#!/bin/bash

#profiler metrics of interest
METRICS=""
#global memory metrics
METRICS+="gld_efficiency,gst_efficiency,gld_transactions,gst_transactions,"
#shared memory metrics
METRICS+="shared_load_transactions_per_request,shared_store_transactions_per_request,shared_efficiency,"
#Occupancy, Branch Divergence
METRICS+="achieved_occupancy,branch_efficiency"

#profiler args
PROFILER_ARGS="--csv" #--log-file $REPORT_FILE2"

#stop=$iKernel
stop=9
iKernel=0
zero=0
while [ $stop -ge 0 ]
do
	echo "Profiling $iKernel.."
    sudo $PROFILER_PATH $PROFILER_ARGS --metrics $METRICS $BIN $iKernel 1>/dev/null 2>>$REPORT_FILE2
	iKernel=$(($iKernel + 1))
	stop=$(($stop - 1))
done
