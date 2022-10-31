# OLFA: An Online Learning Framework for sensor fault analysis in the autonomous cars
This project aims to implement a clustering-based fault classification framework for sensor fault analysis in the autonomous cars. The proposed technique is currently submitted to the IEEE Transactions on Intelligent Transportation Systems.


The Draft_Code.py file implements the proposed OLFA technique into python and users can easily change the name of the datasets in line 25. The construct_W.py and fisher_score.py files are supplemental functions to perform the feature-level analysis on the sensor faults. Since we employed the mutual information score from sklearn package, these two files are not necessary and users can comment out lines 
