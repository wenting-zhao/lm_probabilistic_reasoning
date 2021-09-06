import sys
import math
from collections import Counter
import numpy as np

fn = sys.argv[1]
with open(fn, 'r') as fin:
    lines = fin.readlines()
labels = [float(l) for l in lines if l.strip() != '']
print(np.mean(labels))
tmp = [l for l in labels if l<1/3]
print(len(tmp)/100000)
labels = [str(round(l, 1))[-1] for l in labels]
counter = Counter(labels)
counter = {i: counter[i]/100000 for i in counter}
print([(i, counter[i]) for i in sorted(counter.keys())])

