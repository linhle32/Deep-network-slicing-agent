# Deep network slicing agent

<b>***Update 12/28/2022: My paper has been selected for the Best Paper Award in the DroneCon track!!!</b>

<b>***Update 10/24/2022: My paper has been published in MobiCom 2022 https://dl.acm.org/doi/abs/10.1145/3555661.3560873</b>

### Introduction

5G radio access network (RAN) slicing aims to logically split an infrastructure into a set of self-contained
programmable <b>RAN slices</b>, usually by functionality. Each RAN slice is constituted by various virtual network functions (VNFs) distributed geographically in numerous substrate nodes. Below is an example of three virtual slices placed on top of a physical substrate network

![image](https://user-images.githubusercontent.com/5643444/232635967-67fae4bc-3ebb-4e5b-b02e-3b942dbc65cc.png)

In this research, we develop a deep reinforcement learning based RAN slicing scheme that maximizes the numbers of slices allocated from a demand set. More specifically, at each step of slicing, the current states of the substrate networks requested slices are observed to select a slice to accommodate. The agent consists of a self-attention block to analyze the interrelationship of slices during allocations. Th agent learns to maximize the number of accommodated slices using an explicitly designed reward function. An example of optimal and non-optimal solution is as follows.

![image](https://user-images.githubusercontent.com/5643444/232636059-ed9cfadf-7e7a-4f52-85a0-a96527e1538d.png)

#### Deep agent architecture

![image](https://user-images.githubusercontent.com/5643444/232636081-2c73b037-48c4-4be7-b50d-a82ed368b3ad.png)

### File structure

The use case of this agent is fairly particular, so my priority in packaging it is not high at the moment. The current file structure is as follow 

- <b>utility.py</b>: consists of utility functions and benchmark methods
- <b>network slicing agent.ipynb</b>: building agent and simulation tests

### Libraries

- Python 3
- NumPy
- TensorFlow 2
