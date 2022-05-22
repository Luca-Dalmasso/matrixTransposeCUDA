#!/bin/bash

#check if BINARY exists
if [ ! -f $BIN ]; then
	echo "binary $BIN couldn't be found, compile your application with 'make'"
	exit 1
fi
#check if report dir exists
if [ ! -d $REPORT_DIR ]; then
	echo "report directory doesn't exist, i'm creating it"
	mkdir $REPORT_DIR
fi
#check if log dir exists
if [ ! -d $LOG_DIR ]; then
	echo "log directory doesn't exist, i'm creating it"
	mkdir $LOG_DIR
fi
#every new run requires to backup all logs and reports to tmp directory (this directory should not be versioned)
mkdir -p $TMP_DIR
mv ${REPORT_DIR}/* ${TMP_DIR} 2> /dev/null
mv -f ${LOG_DIR}/* ${TMP_DIR} 2> /dev/null
