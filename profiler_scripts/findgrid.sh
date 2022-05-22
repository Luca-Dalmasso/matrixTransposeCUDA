#!/bin/bash

#best kernel
iKernel=""

declare -a ARGS=(
#		 "8 1"
#		 "8 2"
#		 "8 8"
		 "8 16"
		 "8 32"
		 "8 64"
		 "8 128"
#		 "16 1"
#		 "16 2"
#		 "16 4"
		 "16 8"
		 "16 16"
		 "16 32"
		 "16 64"
#		 "32 1"
#		 "32 2"
		 "32 4"
		 "32 8"
		 "32 16"
	   "32 32"
#		 "64 1"
		 "64 2"
	   "64 4"
		 "64 8"
	   "64 16"
	   "128 1"
	   "128 2"
		 "128 4"
		 "128 8"
		 "256 1"
		 "256 2"
		 "256 4"
		 "512 1"
	   "512 2"
		)

kernelNames[0]="copyRow"
kernelNames[1]="copyCol"
kernelNames[2]="transposeNaiveRow"
kernelNames[3]="transposeNaiveCol"
kernelNames[4]="transposeUnroll4Row"
kernelNames[5]="transposeUnroll4Col"
kernelNames[6]="transposeDiagonalRow"
kernelNames[7]="transposeSmem"
kernelNames[8]="transposeSmemPad"
kernelNames[9]="transposeSmemUnrollPadDyn"
kernelNames[10]="Quit"

echo "Select the kernel"
select opt in ${kernelNames[@]}
do
	case $opt in
		"Quit")
			echo "Goodbye.."
			exit 0
			;;
		*)
			if [ ${#opt} -eq 0 ]
			then
				echo "Invalid selection"
				break
			fi
			id=0
			echo "Peforming grid-level optimization on $opt Kernel"
			#get kernel id
			for str in "${kernelNames[@]}"
			do
				if [ $str == $opt ] 
				then
					iKernel=$id
					break
				fi
				id=$(($id + 1))
			done
			
			for cfg in "${ARGS[@]}"
			do	
				echo "<$iKernel> Block: ($cfg).."
				$BIN $iKernel $cfg 1>> $REPORT_FILE3 2>> $STDERR_LOG
			done
	esac
done



#read report and print
