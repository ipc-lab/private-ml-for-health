#!/bin/sh

#
# Script to build Linux SEAL libraries, python wrapper, and examples
#

# Install binary dependencies
sudo apt-get -qqy update && apt-get install -qqy g++ git make python3 python3-dev
python3-pip libdpkg-perl

cd ~/
git clone https://github.com/Lab41/PySEAL.git

# Build SEAL libraries
cd ~/PySEAL/SEAL/
chmod +x configure
sed -i -e 's/\r$//' configure
./configure
make
export LD_LIBRARY_PATH=~/PySEAL/SEAL/bin:$LD_LIBRARY_PATH

# Build SEAL C++ example
cd ~/PySEAL/SEALExamples
make

# Build SEAL Python wrapper

cd ~/PySEAL/SEALPython
pip3 install --upgrade pip
pip3 install setuptools
pip3 install -r requirements.txt
git clone https://github.com/pybind/pybind11.git
cd ~/PySEAL/SEALPython/pybind11
git checkout a303c6fc479662fd53eaa8990dbc65b7de9b7deb
cd ~/PySEAL/SEALPython
python3 setup.py build_ext -i
export PYTHONPATH=$PYTHONPATH:~/PySEAL/SEALPython:~/PySEAL/bin

# add the following line to your .bashrc file in the home directory

#      export PYTHONPATH=$PYTHONPATH:~/PySEAL/SEALPython:~/PySEAL/bin

# This will allow your python interpreter in the bash terminal to recognize the location of the library everytime you login. Add this path instead to the library paths for your python interpreter in pycharm to make it work in pycharm.


