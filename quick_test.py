import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

with open("data/d1/label", 'r') as f:
    labels = f.readlines()

labels = [float(i) for i in labels]
labels = np.array(labels)
print(labels)
baseline1 = np.random.random(labels.shape) 
baseline2 = np.full(labels.shape, np.mean(labels))
print("random r2", r2_score(labels, baseline1))
print("0.5 r2", r2_score(labels, baseline2))
print("random mae", mean_absolute_error(labels, baseline1))
print("0.5 mae", mean_absolute_error(labels, baseline2))
print("random mse", mean_squared_error(labels, baseline1))
print("mean mse", mean_squared_error(labels, baseline2))
