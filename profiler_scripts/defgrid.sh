#!/bin/bash

stop=0
iKernel=0
while [ $stop == 0 ]
do
	echo "Running $iKernel.."
	./$BIN $iKernel 1>> $REPORT_FILE1 2>> $STDERR_LOG
	stop=$? 
	iKernel=$(($iKernel + 1))
done
kernelName=""
eTime=""
bandW=""
id=0
kernelNames=""
printf "%-20s | %-20s | %-20s |\n" "Kernel Name" "Elapsed time (s)" "bandwidth (MB/s)"
while read line
do
	kernelName=$(echo $line | cut -d',' -f1)
	eTime=$(echo $line | cut -d',' -f3)
	bandW=$(echo $line | cut -d',' -f2)
	printf "%-20s | %-20s | %-20s |\n" $kernelName "$eTime" "$bandW"
	kernelNames[$id]=$kernelName
	id=$(($id + 1))
done < $REPORT_FILE1
kernelNames[$id]="Quit"
