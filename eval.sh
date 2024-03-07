# #!/bin/bash

test_file=$1

if [ ! -n "$test_file" ]; then
    echo 'Error : Expecting file name as input'
    exit 1
fi

if [ -e "$test_file" ]; then
    python KNN.py data.npy $test_file
else
    echo 'Error : File does not exist in the current directory'
fi
