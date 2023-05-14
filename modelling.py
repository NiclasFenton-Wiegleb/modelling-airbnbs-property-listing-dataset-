from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from sklearn.model_selection import GridSearchCV
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import os


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

def save_model(model, hyperparameters: dict, performance_metrics: dict, directory: dir, name: str):
    '''Saves the provided model, hyperparameters and performance_metrics in respective file formats
    in the indicated directory.'''

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


def evaluate_all_models(models_list, directory: str):
    '''This function takes a list of different model classes and a selection of associated
    hyperparameter options, trains all combinations on the dataset, evaluates their performance and
    returns the best overall model and corresponding hyperparameters'''

    best_rmse = None
    best_param_dict = None
    best_model = None
    best_model_type = None
    best_performance_metrics = None

    for d in models_list:

        param_dict, model, performance_dict = tune_regression_model_hyperparameters(**d)

        model_rmse = performance_dict["validation_RMSE"]

        #Save each model to the indicated directory
        file_name = type(model).__name__
        save_model(model, param_dict, performance_dict, directory, file_name)

        print("\nModel Type: ", type(model).__name__)
        print("Hyperparameters: ", param_dict)
        print("Model RMSE: ", model_rmse)

        if best_rmse == None:
            best_rmse = model_rmse
            best_param_dict = param_dict
            best_model = model
            best_model_type = type(model).__name__
            best_performance_metrics = performance_dict

        elif model_rmse < best_rmse:
            best_rmse = model_rmse
            best_param_dict = param_dict
            best_model = model
            best_model_type = type(model).__name__
            best_performance_metrics = performance_dict

        else: continue


    return best_model_type, best_model, best_param_dict, best_performance_metrics

def find_best_model(directory: str):
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
    best_rmse = None
    best_model_name = None
    best_performance_metrics = None

    for tuple in files:

        #Unpack validation RMSE for each model
        performance_dict = tuple[1]
        model_rmse = performance_dict["validation_RMSE"]

        #Unpack model name
        model_name = tuple[0].split("_")[0]

        if best_rmse == None:
            best_rmse = model_rmse
            best_model_name = model_name
            best_performance_metrics = performance_dict

        elif model_rmse < best_rmse:
            best_rmse = model_rmse
            best_model_name = model_name
            best_performance_metrics = performance_dict
    
    model_filename = f"{directory}/{best_model_name}.sav"
    hyperparameters_filename = f"{directory}/{best_model_name}_hyperparameters.json"

    best_model = joblib.load(model_filename)
    best_hyperparameters = json.load(open(hyperparameters_filename))


    return best_model, best_hyperparameters, best_performance_metrics




if __name__ == "__main__":

    directory = "./models/regression"

    best_model, best_hyperparameters, best_performance_metrics = find_best_model(directory)

    # data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")

    # X, y = load_airbnb(data, "Price_Night")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)


    # models_lst = [
    #     {
    #         "model_class": SGDRegressor,
    #         "dataset": (X,y),
    #         "hyperparameters_dict": {"loss": ["squared_error","huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    #                    "shuffle": [True, False]}
    #     },
    #     {
    #         "model_class": RandomForestRegressor,
    #         "dataset": (X, y),
    #         "hyperparameters_dict": {
    #             "n_estimators": [150, 200],
    #             "criterion": ["squared_error", "absolute_error", "friedman_mse"],
    #             "max_depth": [20, 50]
    #         }
    #     },
    #     {
    #         "model_class": GradientBoostingRegressor,
    #         "dataset": (X, y),
    #         "hyperparameters_dict": {
    #             "loss": ["squared_error", "absolute_error", "huber"],
    #             "learning_rate": [0.1, 0.2],
    #             "n_estimators": [150, 200]
    #         }
    #     },
    #     {
    #         "model_class": DecisionTreeRegressor,
    #         "dataset": (X, y),
    #         "hyperparameters_dict": {
    #             "criterion": ["squared_error", "absolute_error", "friedman_mse"],
    #             "max_depth": [20, 50],
    #             "splitter": ["best", "random"]
    #         }
    #     }

    # ]

    # best_model_type, best_model, best_param_dict, best_performance_metrics = evaluate_all_models(models_lst, directory)



