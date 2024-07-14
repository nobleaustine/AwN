
# required libraries 
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pydicom
import os
import nibabel as nib
import pickle



# add a matrix to a hdf5_file or create a dataset with the matrix
def add_matrix(hdf5_file, dataset_name, matrix):

    with h5py.File(hdf5_file, 'a') as file:
        if dataset_name not in file:
            file.create_dataset(dataset_name, data=matrix, maxshape=(None, matrix.shape[1],matrix.shape[2],matrix.shape[3]), chunks=True)
        else:
            file[dataset_name].resize((file[dataset_name].shape[0] + 1), axis=0)
            file[dataset_name][-1, ...] = matrix

# read nii labels from a list of paths and stack to a list  
def read_stack_nii(paths):

    niis = []
    
    for path in paths:
        if os.path.isfile(path):
            nii_slice = nib.load(path)
            niis.append(nii_slice)
        else:
            print(f"Error : {path}")
    
    segments = [nii.get_fdata() for nii in niis]
    labels = [np.transpose(s, (2, 0, 1)) for s in segments]

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

    images = [s.pixel_array for s in slices]
      
    return images

if __name__=="__main__":
  
    with open("list_of_paths.pickle","rb") as file:
        loaded_list = pickle.load(file)


    raw_paths = list(loaded_list["dicom_image_paths"])
    label_paths = list(loaded_list["nii_label_paths"])
    c=1
  
    for raw, label in zip(raw_paths,label_paths):
        if c > 0 :
            print("LAP:",c)
            
            dicom_matrices = read_stack_dicom(raw)
            nii_matrices = read_stack_nii(label)
            
            file_name = f".\\data\\set_{c}.h5"
            
            # adding to hdf5 file
            print("hdf5 file uploading ...")
            add_matrix(file_name, "raw", dicom_matrices)
            add_matrix(file_name, "label", nii_matrices)
        c+=1
