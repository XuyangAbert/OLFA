# OLFA: An Online Learning Framework for sensor fault analysis in the autonomous cars

This project aims to implement a clustering-based fault classification framework for sensor fault analysis in the autonomous cars. The proposed technique is submitted to the IEEE Transactions on Intelligent Transportation Systems and under review. 

The Draft_Code.py file implements the proposed OLFA technique into python and users can easily change the name of the datasets in line 25. The construct_W.py and fisher_score.py files are supplemental functions to perform the feature-level analysis on the sensor faults. Since we employed the mutual information score from sklearn package, these two files are not necessary and users can comment out line 13


## Dependency : Several python packages need to be installed before running the Draft_Code.py file.
1. Numpy
2. Sklearn
3. Pandas
4. Matplotlib
5. Scipy

The code is just a preliminary implementation of OLFA framework and more optimized code will be updated soon.

Along with the python code, we generated six benchmark datasets with different sensor faults using the CARLA simulator. Several examples of the faulty data can be seen below. 

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Latitude-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Longitude-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Heading-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/Velocity-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/AccX-SS.png)

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Plots/AccY-SS.png)

## The output from the OLFA module are:

![alt text](https://github.com/XuyangAbert/OLFA/blob/main/Output.png)
