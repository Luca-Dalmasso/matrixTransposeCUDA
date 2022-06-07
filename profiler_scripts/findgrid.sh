#!/bin/bash

#best kernel
iKernel=""

declare -a ARGS=(
		"7 1"
		"7 7"
		"7 14"
		"7 28"
		"14 1"
		"14 7"
		"14 14"
		"14 28"
		"28 1"
		"28 7" 
		"28 14"
		"28 28"
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
				cat $REPORT_FILE3 | sort -k 4 > ${REPORT_FILE4}
				printf "%-20s | %-20s | %-20s | %20s |\n" "Kernel Name" "Elapsed time (s)" "bandwidth (MB/s)" "Block(x,y)"
				while read line
				do
					kernelName=$(echo $line | cut -d',' -f1)
					eTime=$(echo $line | cut -d',' -f3)
					bandW=$(echo $line | cut -d',' -f2)
					block=$(echo $line | cut -d',' -f6)
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

