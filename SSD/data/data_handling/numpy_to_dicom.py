import numpy as np
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut
import h5py
import nibabel as nib

def numpy_to_nifti(numpy_array, output_file):

    # numpy_array = numpy_array.astype(np.float16)
    numpy_array=np.transpose(numpy_array,(1,2,0))
    img = nib.Nifti1Image(numpy_array, np.eye(4))
    print("shape",img.shape)
    nib.save(img, output_file)

import numpy as np
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian

def numpy_array_to_multiframe_dicom(numpy_array, output_file):
    ds = FileDataset(output_file, {}, file_meta=pydicom.Dataset(), preamble=b"\0" * 128)

    # Set required DICOM attributes
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.ImplementationClassUID = "1.2.276.0.7230010.3.0.3.6.1"

    # Set other necessary attributes
    ds.Modality = 'CT'
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0  # Unsigned integer
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.Rows = numpy_array.shape[1]
    ds.Columns = numpy_array.shape[2]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SpacingBetweenSlices = 1.0

    # Convert the numpy array to a format suitable for DICOM
    pixel_array = numpy_array.astype(np.uint16)
    print("shape1: ",pixel_array.shape)

    # Reshape the pixel array to a single 2D array (frames, rows, columns)
    num_frames = numpy_array.shape[0]
    pixel_array = pixel_array.reshape((num_frames, ds.Rows, ds.Columns))
    print("shape2: ",pixel_array.shape)


    # Set the pixel data
    ds.PixelData = pixel_array

    # Save the DICOM file
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.save_as(output_file)

img_path = ["./1c_image.nii","./1n_image.nii","./5c_image.nii", "./5n_image.nii"]

label_path = ["./1c_label.nii","./5c_label.nii"]

with h5py.File("/cluster/home/austinen/NTNU/DATA/training/heart_1_5/set_1.h5", 'r') as f:
    
    
    for array,img in zip(f["raw"],img_path):
        numpy_to_nifti(array, img)

    for array,img in zip(f["label"],label_path):
        numpy_to_nifti(array, img)
    # numpy_to_nifti(f['label'][0], label_path[0])


# if __name__=="__main__":
