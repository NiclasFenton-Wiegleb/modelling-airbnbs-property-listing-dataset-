from typing import Any
import time
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset
import torch
import tabular_data
import modelling
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import yaml
import itertools, os, json
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
        self.X, self.y = tabular_data.load_airbnb(data, "bedrooms")

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
    def __len__(self):
        return len(self.X)
    
class ANNModel(torch.nn.Module):

    def __init__(self, hidden_layer_width=1, model_depth=1):
        '''Initiate model'''
        super(ANNModel, self).__init__()
        self.hidden_layer_width = hidden_layer_width
        self.model_depth = model_depth
        self.nn_list= [torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width) for i in range(self.model_depth) ]
        self.hidden = torch.nn.ModuleList()
        self.hidden.append(torch.nn.Linear(11, self.hidden_layer_width))
        for i in range(self.model_depth):
            # self.hidden.append(torch.nn.ReLU())
            self.hidden.append(torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width))
        self.hidden.append(torch.nn.Linear(self.hidden_layer_width, 1))

    def forward(self, features):
        '''Defines what the model should do when the __call__ function is called on it'''
        for layer in self.hidden[:-1]:
            features = torch.nn.functional.relu(layer(features))
        output = self.hidden[-1](features)
        return output

def train_model(model_class, dataloader, dataloader_val, batch_size, nn_config, normaliser):
    '''Trains the model and captures the training and validation loss
    via tensorboard'''

    #Initiate timer
    start_timer = time.time()

    #Unpack configurations
    s = Struct(**nn_config)
    optimiser_class = eval(s.optimiser)
    learning_rate = s.learning_rate
    hidden_layer_width = s.hidden_layer_width
    model_depth = s.model_depth
    n_epochs = s.n_epochs

    #Initiate model
    model = model_class(hidden_layer_width, model_depth)
    print(model)

    #Initiate optimizer
    optimizer = optimiser_class(model.parameters(), learning_rate)
    #Initiate loss function
    loss_fn = torch.nn.MSELoss()

    #Initiate tensorboard writer
    layout = {
                "Airbnb_ANN_regression_V01": {
                    "loss": ["Multiline", ["Loss/train", "Loss/validation"]]
                    # "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
                },
            }
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    writer = SummaryWriter("./models/ANN/tensorboard/regression/n_bedrooms/{}".format(current_time))
    writer.add_custom_scalars(layout)

    for epoch in range(n_epochs):
        #Train model
        epoch_train_loss = train_one_epoch(model, epoch, writer, dataloader, optimizer, loss_fn, batch_size, normaliser)
        print(f"Epoch {epoch+1} training loss: ", epoch_train_loss)
        # Validate the model
        epoch_validation_loss = validate_one_epoch(model, loss_fn, epoch, writer, dataloader_val, batch_size, normaliser)
        print(f"Epoch {epoch+1} validation loss: ", epoch_validation_loss)

    writer.flush()
    writer.close()

    stop_timer = time.time()
    training_time = stop_timer - start_timer

    return model, training_time

def train_one_epoch(model, epoch_index, tb_writer, dataloader, optimizer, loss_fn, batch_size, normaliser):

    #Switch the model into training mode
    model.train()

    #Initiate loss values
    running_loss = 0
    last_loss = 0

    n_batches = len(dataloader)

    for i, data in enumerate(dataloader):

        features, labels = data
        features_tensor  = features.float()
        labels_tensor = labels.float()

        features_norm = normaliser(features_tensor)
        y_labels = labels_tensor.reshape([batch_size,1])
        
        #Zero the gradient for each batch
        optimizer.zero_grad()

        #Make predictions
        outputs = model(features_norm)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y_labels)
        loss.backward()

        #Optimise model parameters to reduce loss in next batch
        optimizer.step()

        #Update data
        running_loss += loss.item()

        if i + 1 == n_batches:
            last_loss = running_loss/n_batches
            tb_x = epoch_index + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0

    return last_loss
        
def validate_one_epoch(model, loss_fn, epoch_index, tb_writer, dataloader, batch_size, normaliser):
        
        #Switch model into evaluation mode
        model.eval()
        
        #Validate model for current epoch
        val_running_loss= 0
        last_val_loss = 0
        n_val_batches = len(dataloader)

        for i, val_data in enumerate(dataloader):
                
            X_val, y_val = val_data
            X_val_tensor  = X_val.float()
            y_val_tensor = y_val.float()

            X_val_norm = normaliser(X_val_tensor)


            y_val_labels = y_val_tensor.reshape([batch_size, 1])

            #Evaluating validation error
            outputs = model(X_val_norm)
            mse_val = loss_fn(outputs, y_val_labels)
            val_running_loss += mse_val.item()

            if i + 1 == n_val_batches:
                last_val_loss = val_running_loss/n_val_batches
                tb_x = epoch_index +1
                tb_writer.add_scalar("Loss/validation", last_val_loss, tb_x)
                val_running_loss = 0
        
        return last_val_loss

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_nn_config(yaml_config_file):
    '''Opens yaml file containing the ANN configurations,
    including the optimiser, learning rate, layer width and modal depth and returns
    them as a dictionary.'''
    
    #Open yaml file
    with open(yaml_config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    
    return config_dict

def get_model_inference_latency(model, dataset):

    x, y = dataset

    start_timer = time.time()
    
    y_pred = model(x)

    end_timer = time.time()

    average_time = (end_timer - start_timer)/len(x)
    return average_time

def generate_nn_configs(hyperparameter_dict):
    '''Generates various combinations of coniguration dictionaries
    that can be used to trainn ANNs.'''

    keys, values = zip(*hyperparameter_dict.items())

    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutation_dicts

def find_best_nn(hyperparameter_dict, model_class, dataloader, dataloader_val, batch_size, training_data, validation_data, test_data):
    '''Generates a list of dictionaries of all possible ANN configsurations
    given the provided hyperparameters. A series of model using the given parameters
    are then trained and saved.'''

    normaliser = torch.nn.BatchNorm1d(11)

    #Unpack datasets for performance evaluation
    X_train, y_train = training_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data
    
    X_train_tensor = torch.from_numpy(X_train)
    X_val_tensor = torch.from_numpy(X_validation)
    X_test_tensor = torch.from_numpy(X_test)

    X_train_float = X_train_tensor.float()
    X_val_float = X_val_tensor.float()
    X_test_float = X_test_tensor.float()

    X_train_norm = normaliser(X_train_float)
    X_val_norm = normaliser(X_val_float)
    X_test_norm = normaliser(X_test_float)

    #Generate configurations
    configs_lst = generate_nn_configs(hyperparameter_dict)

    for d in configs_lst:
        #Train model with the current configurations
        model, training_time = train_model(
            model_class= model_class,
            dataloader= dataloader,
            dataloader_val= dataloader_val,
            batch_size= batch_size,
            nn_config= d,
            normaliser= normaliser
        )

        #Evaluate model performance
        y_train_eval = model(X_train_norm)
        y_val_eval = model(X_val_norm)
        y_test_eval = model(X_test_norm)

        performance_dict = {
            "training_RMSE": mean_squared_error(y_train, y_train_eval.detach().numpy(), squared= False),
            "validation_RMSE": mean_squared_error(y_validation, y_val_eval.detach().numpy(), squared= False),
            "test_RMSE": mean_squared_error(y_test, y_test_eval.detach().numpy(), squared= False),
            "training_R^2 score": r2_score(y_train, y_train_eval.detach().numpy()),
            "validation_R^2 score": r2_score(y_validation, y_val_eval.detach().numpy()),
            "test_R^2 score": r2_score(y_test, y_test_eval.detach().numpy()),
            "training_duration": training_time,
            "inference_latency (s)": get_model_inference_latency(model, (X_train_norm, y_train))
            }
        print(performance_dict)
        
        #Save the model
        modelling.save_model(model= model,
                            hyperparameters= d,
                            performance_metrics= performance_dict,
                            is_ANN = True,
                            name = "ANN_lr")

def load_best_model(directory, criterion):
    '''The function loads the performance metrics for each model saved in
    the provided directory and selects the best one based on the validation_RMSE.
    It loads the best model and returns it, along with the corresponding hyperparameters
    and performance metrics.'''

    dir_path = directory

    folders = []

    #iterating through folders in directory and adding 'performance_metrics' dictionary to list of files
    for path in os.listdir(dir_path):
        folders.append(os.path.join(dir_path, path))

    performance_files= []
    hyperparameter_files = []
    model_files = []

    # iterating through files in directory and adding 'performance_metrics' dictionary to list of files
    for dir in folders:
        for path in os.listdir(dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir, path)):

                #Select only files containing the performance metrics for each model
                performance_substring = "performance_metrics"
                hyperparameters_substring = "hyperparameters"
                model_substring = "ANN_lr.pt"

                if performance_substring in path:
                    #Load the performance metrics dictionary and append to list
                    file = json.load(open(os.path.join(dir, path)))
                    performance_files.append(file)
                
                elif hyperparameters_substring in path:
                    #Load the hyperparameter dictionary and append to list
                    file = json.load(open(os.path.join(dir, path)))
                    hyperparameter_files.append(file)
                
                elif model_substring in path:
                    #Load the model and append to list
                    model = torch.load(os.path.join(dir, path))
                    model_files.append(model)

                else: continue

    #Initiatlise variables for model
    best_criterion = None
    best_model_idx = None
    best_performance_metrics = None

    for i, file in enumerate(performance_files):

        #Unpack validation RMSE for each model
        performance_dict = file
        model_criterion = performance_dict[criterion]

        model_idx = i

        if best_criterion == None:
            best_criterion = model_criterion
            best_model_idx = model_idx
            best_performance_metrics = performance_dict

        elif model_criterion > best_criterion:
            best_criterion = model_criterion
            best_model_idx = model_idx
            best_performance_metrics = performance_dict

    best_model = model_files[best_model_idx]
    best_hyperparameters = hyperparameter_files[best_model_idx]
    best_model_file_path = folders[best_model_idx]

    return best_model, best_hyperparameters, best_performance_metrics, best_model_file_path

if __name__ == "__main__":
    
    data = AirbnbNightlyPriceRegressionDataset()

    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size= 0.2)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size= 0.5)

    training_data = list(zip(X_train, y_train))
    validation_data = list(zip(X_validation, y_validation))
    test_data = list(zip(X_test, y_test))

    batch_size = 16

    train_loader = DataLoader(training_data, batch_size= batch_size, shuffle= True, drop_last= True)
    validation_loader = DataLoader(validation_data, batch_size= batch_size, shuffle= True, drop_last= True)
    test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= True, drop_last= True)

    hyperparameter_dict = {
        "optimiser": ["torch.optim.SGD", "torch.optim.Adam"],
        "learning_rate": [0.001, 0.002, 0.0005],
        "hidden_layer_width": [16, 32],
        "model_depth": [15, 30, 50],
        "n_epochs": [30]
    }

    directory = "./models/ANN/regression/n_bedrooms"

    best_model, best_hyperparameters, best_performance_metrics, filepath = load_best_model(directory= directory, criterion= "test_R^2 score")

    print(best_hyperparameters)
    print(best_performance_metrics)
    print(filepath)

    # find_best_nn(
    #     hyperparameter_dict= hyperparameter_dict,
    #     model_class= ANNModel,
    #     dataloader= train_loader,
    #     dataloader_val= validation_loader,
    #     batch_size= batch_size,
    #     training_data= (X_train, y_train),
    #     validation_data= (X_validation, y_validation),
    #     test_data= (X_test, y_test)
    # )