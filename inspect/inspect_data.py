import h5py
file_path = "data/Dataset_Specific_Unlabelled.h5"

with h5py.File(file_path, "r") as f:
    print("Keys in file:")
    for key in f.keys():
        print(key)

    print("/nDataset shapes:")
    for key in f.keys():
        print(key, f[key].shape)

