import pickle
import pandas as pd

with open("../list_of_paths.pickle","rb") as file:
    paths = pickle.load(file)

dicom_paths = list(paths["raw_paths"])
nii_paths = list(paths["label_paths"])

combined_dicom_paths = [element for sublist in dicom_paths for element in sublist]
combined_nii_paths = [element for sublist in nii_paths for element in sublist]
combined_nii_paths = [item for sublist in [[x, x] for x in combined_nii_paths] for item in sublist]


df = pd.DataFrame()
df["dicom_images"] = combined_dicom_paths
df["nii_labels"] = combined_nii_paths

df.to_csv('labeled_data.csv', index=False)


