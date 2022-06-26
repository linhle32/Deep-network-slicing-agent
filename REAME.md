# Deep network slicing agent


### Introduction

5G radio access network (RAN) slicing aims to logically split an infrastructure into a set of self-contained
programmable <b>RAN slices</b>. Each RAN slice is constituted by various virtual network functions (VNFs) distributed geographically in numerous substrate nodes. Below is an example of three virtual slices placed on top of a physical substrate network

![image.png](attachment:image.png)

In this research, we propose a deep reinforcement learning RAN slicing scheme that maximizes the numbers of slices allocated from a demand set. More specifically, at each step of slicing, the current states of the substrate networks requested slices are observed to select a slice to accommodate. The agent consists of a self-attention block to analyze the interrelationship of slices during allocations. Th agent learns to maximize the number of accommodated slices using an explicitly designed reward function. An example of optimal and non-optimal solution is as follows.

![image-2.png](attachment:image-2.png)

#### Deep agent architecture

![image-3.png](attachment:image-3.png)

### File structure

The use case of this agent is fairly particular, so my priority in packaging it is not high at the moment. The current file structure is as follow 

- <b>utility.py</b>: consists of utility functions and benchmark methods
- <b>network slicing agent.ipynb</b>: building agent and simulation tests

### Libraries

- Python 3
- NumPy
- TensorFlow 2
