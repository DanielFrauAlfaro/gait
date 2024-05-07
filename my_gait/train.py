import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Loss function combiend for each task
def multi_task_loss(outputs, labels, heads):

    sum_loss = 0

    for idx_task,task in enumerate(outputs):
        for idx, o in enumerate(task):
            sum_loss = F.cross_entropy(o, labels[idx][idx_task])
    
        
    return sum_loss

# Training function
def train(model, train_loader, heads, num_epochs = 10):

    # Put model in the training device (CPU or GPU if availabel)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store training losses
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        
        # Batch loop
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Put images and labels in the device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)  

            loss = multi_task_loss(outputs, labels, heads)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimizer step
            
            epoch_loss += loss.item()  # Accumulate the loss for the epoch
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')
    del train
    print("Training complete.")

    # Plot the training loss
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model