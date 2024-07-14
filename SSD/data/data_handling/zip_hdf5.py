
# required libraries
import zipfile
import os

# function to zip all files
def convert_zip(zip_file_path,files_to_zip):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            elif os.path.isdir(file):
                for root, dirs, files in os.walk(file):
                    for f in files:
                        file_path = os.path.join(root, f)
                        zipf.write(file_path, os.path.relpath(file_path, file))
    print(f"File '{zip_file_path}' has been created.")

if __name__ == "__main__":

    # List of files to be included in the zip file
    files_to_zip = ["/data/set_1.h5",
                    "/data/set_2.h5",
                    "/data/set_3.h5",
                    "/data/set_4.h5",
                    "/data/set_5.h5"]

    # Name of the zip file to be created
    zip_file_path = '/data/c_1_5.zip'
    # converting to zip
    convert_zip(zip_file_path,files_to_zip)




