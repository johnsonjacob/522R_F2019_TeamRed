# ECEN 522R Fall 2019 Team Red Code Repository (Draft Readme)

## Introduction
THis repository contains the python code used in Dr. Lee's Fall 2019 Self-driving car class.

TODO: Insert picture of car

### File Structure
* Final - Contains the firmware run on the Nvidia Jetson
* yoloServer - Contains the code run on a "cloud"" PC to receive HTTP posts with jpeg images, and returns the decision on stoplight detection.


## Setup Instructions
### yoloServer
#### Dependencies
For the 11/25/19 pass off this was run on ubuntu linux, python3 with mxnet-mkl package installed, not the mxnet-cu**mkl variant. 
