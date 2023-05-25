from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score, accuracy_score
import itertools
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import os
from sklearn import preprocessing
import numpy as np
import torch
from datetime import datetime
from ann import ANNModel



def clf_key_performanc_measures(model, X_train, X_test, X_validation, y_train, y_validation, y_test ):
    '''Returns performance metrics for a given classifier model.'''

    classifier_performance = {
        "f1_score_training": f1_score(y_train, model.predict(X_train), average= "micro"),
        "f1_score_validation": f1_score(y_validation, model.predict(X_validation), average= "micro"),
        "f1_score_test": f1_score(y_test, model.predict(X_test), average= "micro"),
        "precision_score_training": precision_score(y_train, model.predict(X_train), average= "micro"),
        "precision_score_validation": precision_score(y_validation, model.predict(X_validation), average= "micro"),
        "precision_score_test": precision_score(y_test, model.predict(X_test), average= "micro"),
        "recall_score_training": recall_score(y_train, model.predict(X_train), average= "micro"),
        "recall_score_validation": recall_score(y_validation, model.predict(X_validation), average= "micro"),
        "recall_score_test": recall_score(y_test, model.predict(X_test), average= "micro"),
        "accuracy_score_training": accuracy_score(y_train, model.predict(X_train)),
        "accuracy_score_validation": accuracy_score(y_validation, model.predict(X_validation)),
        "accuracy_score_test": accuracy_score(y_test, model.predict(X_test))
        }    
    
    return classifier_performance


def custom_tune_regression_model_hyperparameters(model_class, training_set, validation_set, test_set, param_dict):
    '''The function implements grid search from scratch on the selected model_class using every possible combination 
    of the provided hyperparameters and returns the best hyperparameter set, best model and a dictionary of the model's 
    performance scores. It selects the best model based on the RMSE using the validation dataset. The performance scores 
    include how well it perfoms on the test set.'''

    X_train, y_train = training_set
    X_test, y_test = test_set
    X_validation, y_validation = validation_set

    best_rmse = None
    best_param_dict = None
    best_model = None

    keys, values = zip(*param_dict.items())

    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for d in permutation_dicts:
        model = model_class(**d)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_validation)

        model_rmse = mean_squared_error(y_validation, y_pred, squared= False)

        if best_rmse == None:
            best_rmse = model_rmse
            best_param_dict = d
            best_model = model

        elif model_rmse < best_rmse:
            best_rmse = model_rmse
            best_param_dict = d
            best_model = model
        
        else: continue
    
    performance_dict = {
            "training_RMSE": mean_squared_error(y_train, best_model.predict(X_train), squared= False),
            "validation_RMSE": best_rmse,
            "test_RMSE": mean_squared_error(y_test, best_model.predict(X_test), squared= False),
            "test_R^2 score": r2_score(y_test, best_model.predict(X_test))
            }

    return best_param_dict, best_model, performance_dict


def tune_regression_model_hyperparameters(model_class, dataset, hyperparameters_dict):
    '''This function implements grid search for the best hyperparameters, using GridSearchCV from sklearn'''
    
    X, y = dataset
    model = model_class()
    grid_search = GridSearchCV(model, hyperparameters_dict, return_train_score= True)
    grid_search.fit(X, y)

    best_param_dict = grid_search.best_params_
    best_model = grid_search.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size= 0.2)
    performance_dict = {
            "training_RMSE": mean_squared_error(y_train, best_model.predict(X_train), squared= False),
            "validation_RMSE": mean_squared_error(y_validation, best_model.predict(X_validation), squared= False),
            "test_RMSE": mean_squared_error(y_test, best_model.predict(X_test), squared= False),
            "test_R^2 score": r2_score(y_test, best_model.predict(X_test))
            }

    return best_param_dict, best_model, performance_dict


def tune_classification_model_hyperparameters(model_class, dataset, hyperparameters_dict):
    '''This function implements grid search for the best hyperparameters, using GridSearchCV from sklearn'''
    
    X, y = dataset


    model = model_class()
    grid_search = GridSearchCV(model, hyperparameters_dict, scoring= "accuracy", cv = 5,return_train_score= True)
    grid_search.fit(X, y)

    best_param_dict = grid_search.best_params_
    best_model = grid_search.best_estimator_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size= 0.2)

    performance_dict = clf_key_performanc_measures(best_model,  X_train, X_test, X_validation, y_train, y_validation, y_test)

    return best_param_dict, best_model, performance_dict


def save_model(model, hyperparameters: dict, performance_metrics: dict, label_variable, is_ANN: bool = False, name: str = "Model",directory: dir ="./models/", ):
    '''Saves the provided model, hyperparameters and performance_metrics in respective file formats
    in the indicated directory.'''
    
    #Check if model is ANN
    if is_ANN == True:
        #Create new directory for saving the model with
        # current date and time as name
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_directory = "./models/ANN/regression/{}/{}".format(label_variable, current_time)
        os.mkdir(new_directory)

        #Save the model
        filepath = f"{new_directory}/{name}.pt"
        torch.save(model, filepath)

        #Save the hyperparamters
        hyperparameters_filename = f"{new_directory}/{name}_hyperparameters.json"
        with open(hyperparameters_filename, "w") as f:
            json.dump(hyperparameters, f)
        
        #Saving the performance metrics
        performance_metrics_filename = f"{new_directory}/{name}_performance_metrics.json"
        with open(performance_metrics_filename, "w") as f:
            json.dump(performance_metrics, f)

    if is_ANN == False:
        #Saving the model
        model_filename = f"{directory}/{name}.sav"
        joblib.dump(model, model_filename)

        #Saving the hyperparameters
        hyperparameters_filename = f"{directory}/{name}_hyperparameters.json"
        with open(hyperparameters_filename, "w") as f:
            json.dump(hyperparameters, f)
        
        #Saving the performance metrics
        performance_metrics_filename = f"{directory}/{name}_performance_metrics.json"
        with open(performance_metrics_filename, "w") as f:
            json.dump(performance_metrics, f)


def evaluate_all_models(models_list, tune_function, criterion: str, directory: str):
    '''This function takes a list of different model classes and a selection of associated
    hyperparameter options, trains all combinations on the dataset, evaluates their performance and
    returns the best overall model and corresponding hyperparameters'''

    best_criterion = None
    best_param_dict = None
    best_model = None
    best_model_type = None
    best_performance_metrics = None

    for d in models_list:

        param_dict, model, performance_dict = tune_function(**d)

        model_criterion = performance_dict[criterion]

        #Save each model to the indicated directory
        file_name = type(model).__name__
        save_model(model, param_dict, performance_dict, directory, file_name)

        print("\nModel Type: ", type(model).__name__)
        print("Hyperparameters: ", param_dict)
        print("Model criterion: ", model_criterion)

        if best_criterion == None:
            best_criterion = model_criterion
            best_param_dict = param_dict
            best_model = model
            best_model_type = type(model).__name__
            best_performance_metrics = performance_dict

        elif model_criterion < best_criterion:
            best_criterion = model_criterion
            best_param_dict = param_dict
            best_model = model
            best_model_type = type(model).__name__
            best_performance_metrics = performance_dict

        else: continue


    return best_model_type, best_model, best_param_dict, best_performance_metrics

def find_best_model(directory: str, criterion: str):
    '''The function loads the performance metrics for each model saved in
        the provided directory and selects the best one based on the validation_RMSE.
        It loads the best model and returns it, along with the corresponding hyperparameters
        and performance metrics.'''

    dir_path = directory

    files= []

    #iterating through files in directory and adding 'performance_metrics' dictionary to list of files
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):

            #Select only files containing the performance metrics for each model
            substring = "performance_metrics"
            if substring in path:

                #Load the performance metrics dictionary and append to list
                file = json.load(open(os.path.join(dir_path, path)))
                files.append((path, file))

            else: continue
    
    #Initiatlise variables for model
    best_criterion = None
    best_model_name = None
    best_performance_metrics = None

    for tuple in files:

        #Unpack validation RMSE for each model
        performance_dict = tuple[1]
        model_criterion = performance_dict[criterion]

        #Unpack model name
        model_name = tuple[0].split("_")[0]

        if best_criterion == None:
            best_criterion = model_criterion
            best_model_name = model_name
            best_performance_metrics = performance_dict

        elif model_criterion < best_criterion:
            best_criterion = model_criterion
            best_model_name = model_name
            best_performance_metrics = performance_dict
    
    model_filename = f"{directory}/{best_model_name}.sav"
    hyperparameters_filename = f"{directory}/{best_model_name}_hyperparameters.json"

    best_model = joblib.load(model_filename)
    best_hyperparameters = json.load(open(hyperparameters_filename))


    return best_model, best_hyperparameters, best_performance_metrics


if __name__ == "__main__":

    # data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    
    # X, y = load_airbnb(data, "Category")

    # label_encoder = preprocessing.LabelEncoder()

    # label_encoder.fit(y)
    # y_transform = label_encoder.transform(y)

    # model_list = [
    #     {
    #         "model_class": RandomForestClassifier,
    #         "dataset": (X, y_transform),
    #         "hyperparameters_dict": {
    #             "n_estimators": [150, 200],
    #             "criterion": ["gini", "entropy", "log_loss"],
    #             "max_depth": [20, 50]
    #         }
    #     },
    #     {
    #         "model_class": GradientBoostingClassifier,
    #         "dataset": (X, y_transform),
    #         "hyperparameters_dict": {
    #             "loss": ["log_loss", "deviance", "exponential"],
    #             "learning_rate": [0.1, 0.2],
    #             "n_estimators": [150, 200]
    #         }
    #     },
    #     {
    #         "model_class": DecisionTreeClassifier,
    #         "dataset": (X, y_transform),
    #         "hyperparameters_dict": {
    #             "criterion": ["gini", "entropy", "log_loss"],
    #             "max_depth": [20, 50],
    #             "min_samples_split": [0.1, 0.15, 0.2]
    #         }
    #     },
    #     {   "model_class": LogisticRegression,
    #         "dataset": (X, y_transform),
    #         "hyperparameters_dict":{
    #             "penalty": ["l1", "l2", "elasticnet"],
    #             "dual": [True, False],
    #             "random_state": [0],
    #             "solver": ["lbfgs", "liblinear", "newton-cg"]
    #         }
    #     }
    # ]

    directory = "./models/classification"

    best_model, best_hyperparameters, best_performance_metrics = find_best_model(directory= directory, criterion= "accuracy_score_validation")

    print(type(best_model).__name__)
    print(best_hyperparameters)
    print(best_performance_metrics )
