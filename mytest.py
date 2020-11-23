import os
import re
from sys import path
partern = re.compile(r'feature.*conv.*weight')
print(partern.match('feature00.layers.conv1.weight'))

print(partern.match('feature.conv0.weight'))