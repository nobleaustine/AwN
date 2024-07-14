

# required libraries
import os
import pickle
# import random

# function to get paths of with and without contrast images 1 and 5
# systol and dystol
def get_paths(root_path,round_1):
    
    # two required images type
    img1 =  "kontrast\DICOM\ST00001\SE00001\IM0000"
    img2 =  "kontrast i mageleie\DICOM\ST00001\SE00001\IM0000"
    
    # collect all dicom images and labels
    raw_paths = []
    label_paths = []
    error_paths = []

    round_1 = root_path + round_1

    # taking all rounds and animal paths
    round_paths = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    animal_paths = [d for d in os.listdir(round_1) if os.path.isdir(os.path.join(round_1, d))]
    
    # going through each round and each animal and taking 1st and 5th
    for i,round in enumerate(round_paths):
        print("round: ",i+1)
        for animal in animal_paths:
            sub_raw_paths = []
            sub_label_paths = []
            main = round + "\\" + animal
            c = "\Hjerte med "
            n = "\Hjerte uten "
            if os.path.isfile(main + c + img1 + "1") and os.path.isfile(main + n + img1 + "1") and os.path.isfile(main + c + img1 + "1.nii.gz"):
                sub_raw_paths.append(main + c + img1 + "1")
                sub_raw_paths.append(main + n + img1 + "1")
                sub_label_paths.append(main + c + img1 + "1.nii.gz")

                sub_raw_paths.append(main + c + img1 + "5")
                sub_raw_paths.append(main + n + img1 + "5")
                sub_label_paths.append(main + c + img1 + "5.nii.gz")

            elif os.path.isfile(main + c + img2 + "1") and os.path.isfile(main + n + img2 + "1") and os.path.isfile(main + c + img2 + "1.nii.gz"):
               
                sub_raw_paths.append(main + c + img2 + "1")
                sub_raw_paths.append(main + n + img2 + "1")
                sub_label_paths.append(main + c + img2 + "1.nii.gz")

                sub_raw_paths.append(main + c + img2 + "5")
                sub_raw_paths.append(main + n + img2 + "5")
                sub_label_paths.append(main + c + img2 + "5.nii.gz")
            else:
                error_paths.append(main)

            if not sub_raw_paths:
                print("scan missing")
            else:
              raw_paths.append(sub_raw_paths)
              label_paths.append(sub_label_paths)

    return raw_paths,label_paths,error_paths

if __name__ == "__main__":
    # get raw paths and label
    raw_paths,label_paths,error_paths = get_paths("D:\\Norsvin - CT Segmentation Data","\\AHFP-Scanrunde-1") 


    # print error paths
    print("error paths: ")
    for i in error_paths :
        print(i)
    print("length : ",len(raw_paths))

    # cross checking
    # for i in range(55):
    #     print(i,raw_paths[i],label_paths[i])
    #     print("------------------")

    # saving as a pickle file
    pickle_filename = 'list_of_paths.pickle'
    with open(pickle_filename, 'wb') as file:
        pickle.dump({'dicom_image_paths': raw_paths, 'nii_label_paths': label_paths, 'error_paths': error_paths}, file)
