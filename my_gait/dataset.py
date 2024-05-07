import pickle
import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import math

# Custom Data loader class from Pytorch framework
class BaseGaitDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    # Overwrite function to create the labels and images
    def __getitem__(self, idx):
        image, gait_label, psy_labels = self.data[idx]
        
        # Convert image to torch tensor and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0 
        
        # Concatenates gati and psy labels
        labels = [gait_label] + psy_labels
        
        # Convert label to torch tensor
        labels_tensor = torch.tensor(labels)
        
        return image_tensor, labels_tensor
    

# Custom Gait Dataset class
class GaitDataset():
    def __init__(self, path, types, n_train, n_test, gait_labels, desired_width, desired_height, batch_size):
        
        # Psy labels
        df = pd.read_csv('psimo_reduced/metadata_labels_v3.csv')
        columns = df.columns.tolist()[6:]

        # Obtains the heads / tasks for the network
        self.heads = []
        self.heads.append(len(gait_labels))

        for c in columns:
            df[c] = pd.factorize(df[c])[0]
            self.heads.append(len(df[c].unique().tolist()))

        # Number of training and test individuals
        self.n_train = [str(i) for i in range(n_train)]
        self.n_test = [str(i+n_train) for i in range(n_test)]

        self.df = df
        self.path = path
        self.types= types
        self.gait_labels = gait_labels
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.batch_size = batch_size


    # Method for preparing the dataset
    def prepare_dataset(self, train=True):

        min_length = math.inf
        dataset = []
        if train:
            n_list = self.n_train
        else:
            n_list = self.n_test

        # Two loops, one to check the minimum length and the other to process the images
        for aux in range(2):

            # Loop for each individual
            for num in n_list:

                # Creates the path to that person data
                current_path = os.path.join(self.path, self.types, num)
                
                if not os.path.isdir(current_path):
                    print(f"Directory not found: {current_path}")
                    continue
                
                # Obtains all the subdirectory
                items = os.listdir(current_path)

                # For each subdirectory
                for item in items:

                    # Process each pkl file
                    if True or item.endswith('.pkl'):
                        file_path = os.path.join(current_path, item, item+".pkl")
                        
                        with open(file_path, 'rb') as f:
                            try:
                                # Load sequence
                                images_data = pickle.load(f)

                                # Load labels
                                gait_label = self.gait_labels.index(item.split("_")[-1])
                                psy_label = self.df.iloc[int(num)][6:].tolist()
                                
                                if aux == 1:
                                    # Prepares images
                                    tf_images = []

                                    for idx, image in enumerate(images_data):

                                        # Checks if the sequence is the correct
                                        if idx == min_length: 
                                            break
                                        
                                        # Resize images
                                        image =cv.resize(image, (self.desired_width, self.desired_height))

                                        h, w = image.shape
                                        rgb_image = np.zeros((3, h, w), dtype=np.uint8)

                                        # Copy the grayscale image into all three channels (R, G, B)
                                        rgb_image[0,:,:] = image
                                        rgb_image[1,:,:] = image
                                        rgb_image[2,:,:] = image

                                        tf_images.append(rgb_image)

                                    images_data = np.array(tf_images)
                                
                                if aux == 1:
                                    # Process each image in images_data
                                    dataset.append((images_data, gait_label, psy_label))  
                                else:
                                    if min_length > images_data.shape[0]:
                                        min_length = images_data.shape[0]

                            except Exception as e:
                                print(f"Error loading data from {file_path}: {e}")

        # Creates the data loader
        dataset = BaseGaitDataset(dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader
