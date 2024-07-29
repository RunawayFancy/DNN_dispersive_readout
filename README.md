# Notes for running the scripts

## About

This code is the base code of our work.

> You, Z., Duan, J., Huang, W., Zhang, L., Liu, S., Zhong, Y. and Ian, H., 2023. Neural network based time-resolved state tomography of superconducting qubits. _arXiv preprint arXiv:2312.07958._

## Folder structure for data converting

All raw data, converted data, and train/test result are stored inside `Data/data_from_some_experiment/`.

For raw matlab mat. data, put under `Data/data_from_some_experiment/` and use `mat2txt.m` to convert the mat data to txt., which will be separate in different folder.

The directory of exported data should be in their individual folder respective.
```
-Data
    - data_from_some_experiment
        - train
            - ground
            - excited
        - test
            - ground
            - excited
            - halfpix
            - halfpiy
            - halfpixy
            - halfpiyx
```

Then, use `text_convert.py`, which can automatically convert the data inside all these directory to the correct form. Look up the example in current path setting.

## Python requirements

**We suggest to use [Anaconda](https://www.anaconda.com/) to manage your virtual environment on your pc.**

* Python version

```
Python version >= 3.10
```

* Requirement package
  
Use `requirement.txt` to install all require pipy package
```
pip install -r requirement.txt
```
