# [MECH439] Introduction to Robotics
project for a POSTECH MECH439 (Introduction to Robotics) lecture

## Requirements
1. Create conda environment
```shell
$ conda create -n <ENV_NAME> python==3.10 # create virtual environment
$ conda activate <ENV_NAME>
$ pip install -r requirements.txt # install dependancies from requirements.txt
```

2. Install Pinocchio (Regid Body Dynamics Library)
```shell
$ conda install pinocchio -c conda-forge    # For Windows
$ pip install pin    # For Ubuntu
```