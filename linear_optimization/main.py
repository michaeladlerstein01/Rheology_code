from helper import *
import torch 
import numpy as np 
import matplotlib.pyplot as plt 


mat1 = torch.rand((100 , 500))
mat2 = torch.rand((100 , 500))


print(torch.sum(mat1 + mat2))


