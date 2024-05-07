from my_gait.dataset import *
from my_gait.model import *
from my_gait.test import *
from my_gait.train import *

# Path to files
path = "./psimo-reduced-64-64-pkl/"
types = "silhouettes/"
label_path = "psimo_reduced/metadata_labels_v3.csv"

n_train = 9     # Number of training samples
n_test = 1      # Number of test samples
batch_size = 16
dataset = []  # List to store processed images and labels

# Final width and heigth of images
desired_width = 224
desired_height = 224

# Labels for gait
gait_labels = ["bg", "cl", "nm", "ph", "txt", "wsf", "wss"]

# Dataset object
dataset = GaitDataset(path, types, n_train, n_test, gait_labels, desired_width, desired_height, batch_size)

# Prepare training and test dataset
train_dataset = dataset.prepare_dataset()


# Train model
model = ResNetLSTM(dataset.heads)
model = train(model, train_loader=train_dataset, heads=dataset.heads)

del train_dataset
test_dataset = dataset.prepare_dataset(train = False)
# Evaluate model
__ = evaluate_model(model, test_dataset)