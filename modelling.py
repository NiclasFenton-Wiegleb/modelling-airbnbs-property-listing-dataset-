from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools
from sklearn.model_selection import GridSearchCV

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

    performance_dict = grid_search.cv_results_
    best_param_dict = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_param_dict, best_model, performance_dict



if __name__ == "__main__":

    data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")

    X, y = load_airbnb(data, "Price_Night")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size= 0.2)

    hyperparameters = {"loss": ["squared_error","huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
                       "shuffle": [True, False]}

    best_param_dict, model, performance_dict = tune_regression_model_hyperparameters(SGDRegressor, (X, y),
                                                                                                hyperparameters)
    print(best_param_dict)
    print(model.score(X_test, y_test))
    print(pd.DataFrame(performance_dict))



