# OLFA: An Online Learning Framework for sensor fault analysis in the autonomous cars

This project aims to implement a clustering-based fault classification framework for sensor fault analysis in the autonomous cars. The proposed technique is submitted to the IEEE Transactions on Intelligent Transportation Systems and under review. (Modifications will be added later to address any possible issues of the proposed technique.)

In "Main_code" folder, the Draft_Code_SS.py/Draft_Code_MS.py file implements the proposed OLFA technique in python and users can easily change the name of the datasets in line 25. The construct_W.py and fisher_score.py files are supplemental functions to perform the feature-level analysis on the sensor faults. Since we employed the mutual information score from sklearn package, these two files are not necessary and users can comment out line 13.

In "Plots" folder, a visualization of the injected single-sensor faults with three fault models is displayed.


## Several python packages need to be installed before running the Draft_Code.py file.
1. Numpy
2. Sklearn
3. Pandas
4. Matplotlib
5. Scipy

## Note: The code is just a preliminary implementation of OLFA framework and more optimized code will be updated soon.

Along with the python code, we generated six benchmark datasets with different sensor faults using the CARLA simulator. Several examples of the faulty data can be seen below. 

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Latitude-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Longitude-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Heading-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Velocity-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/AccX-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/AccY-SS.png)

## The workflow of the OLFA module can be visualized below:

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Output.png)

## Note:
This work is submitted to the IEEE Transactions on Intelligent Transportation Systems and is currently under review. Modification will be constantly updated to address possible issues raised in the future.


