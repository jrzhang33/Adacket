# Adacket: ADAptive Convolutional KErnel Transform for Multivariate Time Series Classification



## Description

This repo contains the PyTorch implementation for paper: Adacket: ADAptive Convolutional KErnel Transform for Multivariate Time Series Classification (Accepted at ECML-PKDD 2023)



## Training Adacket

1. You can obtain UEA datasets from the UCR official website [[Welcome to the UCR Time Series Classification/Clustering Page](http://www.cs.ucr.edu/~eamonn/time_series_data/)] and place them in datasets. At the same time, the UCR official website displays the attributes of all MTSC datasets (number of samples, length, and channel number). Here we take the first UEA dataset as an example.
2. Run Main.py. Here,you can modify the args. dataset to run the dataset with the specified name.
3. After running, you will obtain [name, feature quantities, parameter quantities, training accuracy, and test accuracy]. 



## Adacket  Results

We run all experienments  on one GeForce RTX 3090 Ti GPU , running 64-bit Linux 5.15.0-56-generic.

We have shown the results of Adacket in terms of accuracy, parameter numbers, and memory usage, and compared them with Rocket, InceptionTime, TapNet, and ResNet. The results show that Adacket achieves the best ranking in terms of accuracy, parameter consumption, and memory usage, which means that Adacket consumes fewer parameters and memory while achieving higher accuracy.

Here is the results: https://github.com/jrzhang33/Adacket/blob/main/results/result_acc_params_memory_30UEA.xlsx.



## Appendix

More detailed supplementary material is shown at https://github.com/jrzhang33/Adacket/blob/main/Adacket_Appendix.pdf
