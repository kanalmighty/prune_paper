import os
import re
from sys import path
conv_pattern = re.compile(r'feature.*conv.*weight')
bias_pattern = re.compile(r'feature.*conv.*bias')
norm_pattern = re.compile(r'feature.*norm.*')
linear1_pattern = re.compile(r'classifier.*.*')
print(conv_pattern.match('feature.0.layers.conv1.weight'))
print(bias_pattern.match('feature.3.layers.norm2.bias'))
print(norm_pattern.match('feature.3.layers.norm2.weight'))
print(linear1_pattern.match('classifier.weight'))


# print(partern.match('feature.conv0.weight'))