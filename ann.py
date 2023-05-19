from typing import Any
import pandas as pd
from torch.utils.data import Dataset
import torch
import tabular_data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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
        self.linear_layer = torch.nn.Linear(9, 16, bias= False)

    def forward(self, features):
        return self.linear_layer(features)

    
    def train(self, dataloader, n_epochs):

        for epoch in range(n_epochs):
            for batch in dataloader:
                example = next(iter(train_loader))
                features, labels = example

                features = features.reshape(len(labels), -1)
                features = features.type('torch.FloatTensor')

                print(self(features))
                break


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

    model.train(training_data, 2)
