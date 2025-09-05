# DL-DualTENGSensor-SP
Deep Learning-based Dual-reference Triboelectric Sensor for Direct Surface Potential Prediction

This repository shows how to load the data and train the model associated with the paper “Deep Learning-based Dual-reference Triboelectric Sensor for Direct Surface Potential Prediction”. This code was initially performed in MATLAB, but it is also possible to access information in Python. In the MATLAB files (.mat), this data is stored in a cell according to the periodic and repetitive experiments. The raw data is also available in Excel format (.csv). 


1. The SP.csv file includes the list of materials, index numbers, average surface potential values, and standard deviation. 

2. The data associated with dataCyc_h2050.mat is as follows. 
- TrainSet: It includes the material index used to train the model, namely, PTFE, Kapton, PET, PMMA, Paper, Glass, and Nylon.

- TestSet: It includes the material index used to test the model, namely,  PVDF, Al, and TPU

- Xraw: The raw data sample consists of two-channel time-series voltage signals measured from PDMS and PEI(b)-PDMS sensing layers, with a shape of 120 × 2, sampled at 500 Hz. 

- XTrain: The training input data consists of differences between adjacent elements of time-sequential measurements within trainSet.

- YTrain: The average surface potential corresponds to the material used for XTrain.

- XTestAL (XTestTPU, XTestPVDF): The testing input data consists of differences between adjacent elements of time-sequential measurements for Al, TPU, and PVDF. 

- YTestAL (YTestTPU, YTestPVDF): The average surface potential data corresponds to the testing data of Al, TPU, and PVDF. 

3. The TrainTCN.m code includes how to load the data, normalize the input, define, train, and test the model. 
