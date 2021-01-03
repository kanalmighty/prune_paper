import os
import re
from sys import path
import torch
import torch.nn as nn
input = torch.ones(3,3)
weight = torch.ones(3)
bias = torch.ones(1,3)
print(torch.addmm(bias, input, weight.t()))

# print(partern.match('feature.conv0.weight'))