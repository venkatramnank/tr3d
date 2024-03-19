import os
import h5py
import random
import pickle
import sys


def process_directory(directory):
    # List to store [index, file name, frame number]
    file_info_list = []
    img_idx = 0
    # Iterate and count frames in HDF5 files
    
    for root, dirs, files in os.walk(directory):
        for file in files[:10]:
            if file.endswith('.hdf5') and 'towers' in file:
                file_path = os.path.join(root, file)
                with h5py.File(file_path, 'r') as f:
                    frame_count = len(f['frames'].keys())
                    for frame_id in range(frame_count):
                        file_info_list.append([img_idx, file_path, '{:04d}'.format(frame_id)])
                        img_idx += 1
    return file_info_list


def main():
    if len(sys.argv) < 2:
        print("Usage: python store_physion_onthefly_data.py directory1 directory2 directory3 ...")
        sys.exit(1)

    directories = sys.argv[1]
    file_info_list = process_directory(directories)

    # Randomly shuffle the list
    random.shuffle(file_info_list)

    # Split the list into train and validation sets
    split_index = int(0.95 * len(file_info_list))
    train_data = file_info_list[:split_index]
    val_data = file_info_list[split_index:]
    parent_directory = os.path.dirname(directories)
    # Save train and validation sets as pickle files
    with open(parent_directory + '/train_onthefly_data_small.pkl', 'wb') as train_file:
        pickle.dump(train_data, train_file)

    with open(parent_directory + '/val_onthefly_data_small.pkl', 'wb') as val_file:
        pickle.dump(val_data, val_file)

if __name__ == "__main__":
    main()
