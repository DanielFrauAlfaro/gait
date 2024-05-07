import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Evaluate model function
def evaluate_model(model, data_loader):

    # Put model in the test device (CPU or GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()  
    all_labels = []
    all_predictions = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():

        # Batch loop  
        for images, labels in data_loader:

            # Put images and labels in the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # List of all predictions
            predictions = [[] for _ in range(images.shape[0])]

            # For each output and for each task obtain the maximum value (predicted)
            for idx_task,task in enumerate(outputs):            
                for idx, o in enumerate(task):
                    predictions[idx].append(torch.argmax(o).item())

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions)

    accuracy = []
    precision = []
    recall = []
    f1 = []
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate evaluation metrics for each task
    for i in range(len(all_predictions[0])):
        accuracy.append(accuracy_score(all_labels[:][i], all_predictions[:][i]))
        precision.append(precision_score(all_labels[:][i], all_predictions[:][i], average='macro', zero_division=0))
        recall.append(recall_score(all_labels[:][i], all_predictions[:][i], average='macro', zero_division=0))
        f1.append(f1_score(all_labels[:][i], all_predictions[:][i], average='macro', zero_division=0))


    print(f"Accuracy: {accuracy}\n\n\n Precision: {precision}\n\n\n Recall: {recall}\n\n\n F1-score: {f1}")

    return accuracy, precision, recall, f1


