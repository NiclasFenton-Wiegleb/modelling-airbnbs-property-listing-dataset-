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

'''

TODO: Add visual investigation and use it to justify the selection of models evaluated and associated parameters.

- Weigh up pros and cons of different models
- Talk about how to avoid over and under fitting and various issues related to the models
- Why use RMSE as a criterion?

- Notebook?

'''

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
- Test R<sup>2</sup> : 0.999

#CHECK BEST MODEL -> error in code selecting best model

## Classification Models

As our next challenge, we aim to train a model that can predict the category of a listing using its various attributes. As there is a set list of categories available, this is a multiclass problem - therefore, requiring a classification model to solve. It's also important to note tha the categories have to be encoded using the LabelEncoder() provided by sklearn. This transforms the different classes into integers that can be passed into an array, rather than strings. The inverse_transform() method can be used to translate the predicet result back into the original class.

The grid search for obtaining the best model and corresponding hyperparamters can be done in a similar manner to the regression models mentioned above, with some tweaks to the existing functions. This includes making them more general so that, for example, they can take in different criterions to evaluate the trained models against. Above we used the RMSE of the validation set, which would not make much sense for a classification problem. Hence, we will be using the accuracy of the model to predict classes in the validation set as a metric in this section.


'''

TODO: Add visual investigation and use it to justify the selection of models evaluated and associated parameters.

- Talk about how to avoid over and under fitting and various issues related to the models
- Why use accuracy as a criterion?
- investigate/visualise performance of models of each model_class

- Notebook?


'''

This is the list of models and hyperparameters that were evaluated:

```
model_list = [
        {
            "model_class": RandomForestClassifier,
            "dataset": (X, y_transform),
            "hyperparameters_dict": {
                "n_estimators": [150, 200],
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [20, 50]
            }
        },
        {
            "model_class": GradientBoostingClassifier,
            "dataset": (X, y_transform),
            "hyperparameters_dict": {
                "loss": ["log_loss", "deviance", "exponential"],
                "learning_rate": [0.1, 0.2],
                "n_estimators": [150, 200]
            }
        },
        {
            "model_class": DecisionTreeClassifier,
            "dataset": (X, y_transform),
            "hyperparameters_dict": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [20, 50],
                "min_samples_split": [0.1, 0.15, 0.2]
            }
        },
        {   "model_class": LogisticRegression,
            "dataset": (X, y_transform),
            "hyperparameters_dict":{
                "penalty": ["l1", "l2", "elasticnet"],
                "dual": [True, False],
                "random_state": [0],
                "solver": ["lbfgs", "liblinear", "newton-cg"]
            }
        }
    ]

```

All of the best performing models for each model class were saved to the directory ./models/classification, along with their hyperparameters and performance metrics.

In this scenario a DecisionTreeClassifier model performed best with the following hyperparameters:

criterion: entropy
max_depth: 20
min_samples_split: 0.15

The corresponding performance metrics are listed below:

f1_score_training: 0.39759036144578314
f1_score_validation: 0.4117647058823529
f1_score_test: 0.48484848484848486
precision_score_training: 0.39759036144578314
precision_score_validation: 0.4117647058823529
precision_score_test: 0.48484848484848486
recall_score_training: 0.39759036144578314
recall_score_validation: 0.4117647058823529
recall_score_test: 0.48484848484848486
accuracy_score_training: 0.39759036144578314
accuracy_score_validation: 0.4117647058823529
accuracy_score_test: 0.48484848484848486

#CHECK BEST MODEL -> error in code selecting best model

## ANN Regression Model

'''
Update documentation:
    - model architecture
    - how it was used
    - screenshots of tensorboard graphs
    - show training for all models then focus in on best paramterised model
    - use this link (https://alexlenail.me/NN-SVG/) to generate diagram of ANN
    - Limitations of ANN
    - Comparision to ML regression model

'''