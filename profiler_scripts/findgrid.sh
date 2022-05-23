#!/bin/bash

#best kernel
iKernel=""

declare -a ARGS=(
		 "8 1"
		 "8 2"
		 "8 8"
		 "8 16"
		 "8 32"
#		 "8 64"
#		 "8 128"
#		 "16 1"
#		 "16 2"
#		 "16 4"
		 "16 8"
		 "16 16"
#		 "16 32"
#		 "16 64"
#		 "32 1"
#		 "32 2"
		 "32 4"
		 "32 8"
		 "32 16"
	     "32 32"
#		 "64 1"
		 "64 2"
#	     "64 4"
#		 "64 8"
#	     "64 16"
	     "128 1"
	     "128 2"
#		 "128 4"
#		 "128 8"
		 "256 1"
		 "256 2"
#		 "256 4"
#		 "512 1"
#	     "512 2"
		)
		
echo "Select the kernel"
select opt in ${kernelNames[@]}
do
	case $opt in
		"Quit")
			if [ -f $REPORT_FILE3 ]
			then
				echo "Results.."
				kernelName=""
				eTime=""
				bandW=""
				block=""
				cat $REPORT_FILE3 | sort -k 3 > ${REPORT_FILE4}
				printf "%-20s | %-20s | %-20s | %20s |\n" "Kernel Name" "Elapsed time (s)" "bandwidth (MB/s)" "Block(x,y)"
				while read line
				do
					kernelName=$(echo $line | cut -d',' -f1)
					eTime=$(echo $line | cut -d',' -f3)
					bandW=$(echo $line | cut -d',' -f2)
					block=$(echo $line | cut -d',' -f5)
					printf "%-20s | %-20s | %-20s | %-20s|\n" $kernelName "$eTime" "$bandW" "$block"
				done < $REPORT_FILE4
			fi
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

