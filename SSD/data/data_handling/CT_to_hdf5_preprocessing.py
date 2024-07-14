
# required libraries 
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pydicom
import os
import nibabel as nib
import pickle

# function to pad the individual label and raw image and stack them to a single array
# def pad_and_stack(pixel_arrays):

#     # Find the maximum size along the first axis
#     max_size = max(arr.shape[0] for arr in pixel_arrays)

#     # Create a list to store padded arrays
#     padded_arrays = []

#     # Pad each array to the maximum size along the first axis
#     for arr in pixel_arrays:
#         # Calculate padding 
#         pad_width = ((0, max_size - arr.shape[0]), (0, 0), (0, 0))

#         # Pad or crop along the first axis
#         padded_array = np.pad(arr, pad_width, mode='constant', constant_values=0)

#         # Append the padded or cropped array to the list
#         padded_arrays.append(padded_array)

#     # Stack the padded or cropped arrays along the first axis
#     stacked_array = np.stack(padded_arrays, axis=0)

#     return stacked_array

# add a matrix to a hdf5_file or create a dataset with the matrix
def add_matrix(hdf5_file, dataset_name, matrix):
    with h5py.File(hdf5_file, 'a') as file:
        # Create or open the dataset
        if dataset_name not in file:
            file.create_dataset(dataset_name, data=matrix, maxshape=(None, matrix.shape[1],matrix.shape[2],matrix.shape[3]), chunks=True)
        else:
            # Resize the dataset to accommodate the new matrix
            file[dataset_name].resize((file[dataset_name].shape[0] + 1), axis=0)
            file[dataset_name][-1, ...] = matrix

# read nii labels from a list of paths and stack to a list  
def read_stack_nii(paths):

    niis = []
    # check for the path and load and append the label else print error path
    for path in paths:
        if os.path.isfile(path):
            nii_slice = nib.load(path)
            niis.append(nii_slice)
        else:
            print(f"Error : {path}")
    # stack all images and transpose to align with image 
    segments = [nii.get_fdata() for nii in niis]
    
    labels = [np.transpose(s, (2, 0, 1)) for s in segments]
    # labels = pad_and_stack(segments)

    return labels

# read dicom images from a list of paths and stack to a list
def read_stack_dicom(paths):

    slices = []

    for path in paths:
        if os.path.isfile(path):
            dicom_slice = pydicom.read_file(path)
            slices.append(dicom_slice)
        else:
            print("Error: ",path)

    # sizes =[(s.pixel_array).shape for s in slices]
    # types = [(s.pixel_array[250][250][250]) for s in slices]
    # print(sizes)
    # print(types)
    images = [s.pixel_array for s in slices]
    
    # Convert to numpy
    # scans = np.stack([s.pixel_array for s in slices])
    # scans = pad_and_stack(slices)

    # scans = scans.astype(np.int16)
    # converting to range within 0-1
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # image_shape = scans[0].shape
    # images = np.array([(scaler.fit_transform(img.reshape(-1,1))).reshape(image_shape) for img in scans], dtype=np.float32)

    
    return images

if __name__=="__main__":

    # open the pickle file of list of list of paths
    with open("list_of_paths.pickle","rb") as file:
        loaded_list = pickle.load(file)

    # getting images and labels
    raw_paths = list(loaded_list["raw_paths"])
    label_paths = list(loaded_list["label_paths"])
    c=1
    # converting list of paths into images and labels and storing as hdf5 file
    for raw, label in zip(raw_paths,label_paths):
        if c > 0 :
            print("LAP:",c)
            # print(raw)
            # print(label)

            dicom_matrices = read_stack_dicom(raw)
            nii_matrices = read_stack_nii(label)
            
            file_name = f"C:\\\NOBLEAUSTINE\\GitWorld\\MedSegCon\\data\\set_{c}.h5"
            
            # adding to hdf5 file
            print("hdf5 file uploading ...")
            add_matrix(file_name, "raw", dicom_matrices)
            add_matrix(file_name, "label", nii_matrices)
        c+=1
