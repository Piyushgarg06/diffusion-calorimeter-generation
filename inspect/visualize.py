import matplotlib.pyplot as plt
import h5py
f = h5py.File("data/Dataset_Specific_Unlabelled.h5","r")
data = f["jet"]

sample = data[0]

plt.imshow(sample[:,:,0])
plt.title("Channel 0")
plt.colorbar()
plt.show()
import numpy as np

print("Minimum value:", np.min(sample))
print("Maximum value:", np.max(sample))
print("Average value:", np.mean(sample))

non_zero = np.count_nonzero(sample)
total = sample.size

print("Non-zero pixels:", non_zero)
print("Total pixels:", total)
print("Fraction non-zero:", non_zero / total)