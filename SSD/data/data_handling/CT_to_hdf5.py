
# required libraries 
import h5py
import numpy as np
import pydicom
import os
import nibabel as nib
import pickle

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
            
            file_name = f"./data/set_{c}.h5"
            
            # adding to hdf5 file
            print("hdf5 file uploading ...")
            with h5py.File(file_name, 'a') as file:   
                file.create_dataset("c1", data=dicom_matrices[0])
                file.create_dataset("n1", data=dicom_matrices[1])
                file.create_dataset("c5", data=dicom_matrices[2])
                file.create_dataset("n5", data=dicom_matrices[3])

                file.create_dataset("l1", data=dicom_matrices[0])
                file.create_dataset("l5", data=dicom_matrices[1])   
        c+=1
