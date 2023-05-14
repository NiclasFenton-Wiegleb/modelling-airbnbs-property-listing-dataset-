# Modelling Airbnb's Property Listing Dataset

## Description

In this project the challenge is to create several Machine Learning and Deep Learning models that are trained on data and fulfill functions relevant to the Airbnb application. Please see below for a list of relevant dependancies and technologies used in throughout the project.

Dependencies:

- Pandas
- Numpy
- Sklearn

Technologies:

- Python
- Sci-kit learn

## Cleaning the Data

The original data is shown in the file "listing.csv" and requires some cleaning. The following actions are completed by passing the dataframe of the original file into the relevant functions of the DataCleaning() class shown in tabular_data.py

- Removing rows with missing ratings for any of the ratings columns
- Combining the list of strings in the description column into a single, clean string
- Setting the default value to one for any missing values in the guests, beds, bathrooms or bedrooms columns.

*Pre-processing*: In order to utilise the clean data for training, it is passed to the load_airbnb(df, labels) which returns a tuples in the format (features, labels) for each example in the dataset.

## Regression Models

The first challenge in this scenario is to find the best regression model that can fit the data - processed in the previous step - and predict the price per night of a listing based on the different ratings and stats (e.g. number of bedrooms) for each listing. To do this we implement grid search. The file modelling.py contains a function that implements grid search from scratch (custom_tune_regression_model_hyperparameters()) and a second that makes use of the sci-kit learn GridSearchCV class (tune_regression_model_hyperparameters()).

We provide the function with a list of dictionaries, each specifying a model class for our regression model that we want to train, as well as a range of hyperparameters associated with the model. The function then goes through all possible hyperparameter configurations for each model and saves the model of each type to the provided directory. We then return the overall best model, hyperparameters and the performance metrics of the model.

This is the list of models and hyperparameters that were evaluated:

```
    models_lst = [
        {
            "model_class": SGDRegressor,
            "dataset": (X,y),
            "hyperparameters_dict": {"loss": ["squared_error","huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
                       "shuffle": [True, False]}
        },
        {
            "model_class": RandomForestRegressor,
            "dataset": (X, y),
            "hyperparameters_dict": {
                "n_estimators": [150, 200],
                "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                "max_depth": [20, 50]
            }
        },
        {
            "model_class": GradientBoostingRegressor,
            "dataset": (X, y),
            "hyperparameters_dict": {
                "loss": ["squared_error", "absolute_error", "huber"],
                "learning_rate": [0.1, 0.2],
                "n_estimators": [150, 200]
            }
        },
        {
            "model_class": DecisionTreeRegressor,
            "dataset": (X, y),
            "hyperparameters_dict": {
                "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                "max_depth": [20, 50],
                "splitter": ["best", "random"]
            }
        }
    ]
```

The find_best_model function goes through all the models in a specified directory and loads the best one.

*Note*: For consistancy, throughout the evaluation process, the Root Mean Squared Error (RMSE) for a validation dataset is used for comparing model performance.

The best model found within the list above was of the DecisionTreeRegressor class.

With the hyperparameters:

- criterion: absolute error
- max depth: 50
- splitter: random

And the performance metrics:

- Training RMSE: 5.923
- Validation RMSE: 0.0
- Test RMSE: 4.826
- Test R<sup>2<sup> : 0.999
