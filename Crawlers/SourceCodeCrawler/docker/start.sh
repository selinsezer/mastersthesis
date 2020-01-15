#!/bin/bash

cd /app
echo "Starting python"
python -V
python ./main.py --range 0-5
echo "goodbye"