from typing import Any
import pandas as pd
from torch.utils.data import Dataset
import torch
import tabular_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torcheval.metrics import MeanSquaredError
from torch.utils.tensorboard import SummaryWriter

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
        self.X, self.y = tabular_data.load_airbnb(data, "Price_Night")


    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        return len(self.X)
    
class ANNModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(9, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 16)
        )

    def forward(self, features):
        predictions = self.linear_relu_stack(features)
        return predictions

    
    def train(self, dataloader, n_epochs, dataloader_val):

        #Initiate optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        #Initiate loss function
        loss_fn = torch.nn.MSELoss()

        #Determine number of batches in the dataloader
        n_batches = [i for i, batch in enumerate(dataloader)]
        max_batch = max(n_batches)+1
        #Initiate batch counting
        current_batch = 0

        #Initiate loss values
        running_loss = 0
        last_loss = 0

        #Initiate tensorboard writer
        writer = SummaryWriter("./models/ANN/tensorboard/regression")

        for epoch in range(n_epochs):
            #Train model
            for i, data in enumerate(dataloader):

                features, labels = data
                features_tensor  = torch.tensor(features, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.float32)

                #Zero the gradient for each batch
                optimizer.zero_grad()

                #Make predictions
                outputs = self(features_tensor)

                #Compute the loss and its gradients
                loss = loss_fn(outputs, labels_tensor)
                loss.backward()


                #Optimise model parameters to reduce loss in next batch
                optimizer.step()

                #Update data
                running_loss += loss.item()
                current_batch += 1

                if current_batch == max_batch:
                    last_loss = running_loss/max_batch
                    print(f"Epoch {epoch} loss: ", last_loss)

            current_batch = 0
            running_loss = 0
            
            #Validate model for current epoch
            n_batches_val = [i for i, batch in enumerate(dataloader_val)]
            max_batch_val = max(n_batches_val)+1
            
            current_batch_val = 0
            val_running_error = 0
            last_val_error = 0

            for i, val_data in enumerate(dataloader_val):
                    
                X_val, y_val = val_data
                X_val_tensor  = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                #Evaluating validation error
                output = self(X_val_tensor)

                i, y_pred = torch.max(output, 1)

                #Initiate evaluation metric
                metric = MeanSquaredError()
                metric.update(y_pred, y_val_tensor)
                val_running_error += metric.compute()
                current_batch_val += 1

                if current_batch_val == max_batch_val:
                    last_val_error = val_running_error/max_batch_val

            print(f"Epoch {epoch} validation error: ", last_val_error)

            current_batch_val = 0
            val_running_error = 0


if __name__ == "__main__":
    
    data = AirbnbNightlyPriceRegressionDataset()

    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size= 0.2)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size= 0.2)

    training_data = list(zip(X_train, y_train))
    validation_data = list(zip(X_validation, y_validation))
    test_data = list(zip(X_test, y_test))

    batch_size = 16

    train_loader = DataLoader(training_data, batch_size= batch_size, shuffle= True)
    validation_loader = DataLoader(validation_data, batch_size= batch_size, shuffle= True, drop_last= True)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= True, drop_last= True)

    model = ANNModel()

    model.train(training_data, 3, validation_loader)

    # for i, data in enumerate(train_loader):
    #         #TODO create training loop that trains model, evaluates it on validation set and optimises

    #         features, labels = data
    #         #Reshape and format features to fit input layer of model
    #         label_len = [i for i, labels in enumerate(labels)]
    #         # features = features.reshape(max(label_len), -1)
    #         # features = features.type('torch.FloatTensor')
    #         # print(features.dtype)
    #         # print(features.shape)
    #         print(label_len)
    #         break


    


