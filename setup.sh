#!/usr/bin/env bash

# install venv (in case it is not installed)
sudo apt-get update
sudo apt-get install python3-venv

#create virtual environment
python -m venv VA3_env

#activate virtual environment
source ./VA3_env/bin/activate

#install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# deactivate environment
deactivate
