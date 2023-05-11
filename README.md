# Modelling Airbnb's Property Listing Dataset

## Description

In this project the challenge is to create several Machine Learning and Deep Learning models that are trained on data and fulfill functions relevant to the Airbnb application. Please see below for a list of relevant dependancies and technologies used in throughout the project.

Dependencies:

- Pandas
- Numpy

Technologies:

- Python

## Cleaning the Data

The original data is shown in the file "listing.csv" and requires some cleaning. The following actions are completed by passing the dataframe of the original file into the relevant functions of the DataCleaning() class shown in tabular_data.py

- Removing rows with missing ratings for any of the ratings columns
- Combining the list of strings in the description column into a single, clean string
- Setting the default value to one for any missing values in the guests, beds, bathrooms or bedrooms columns.

*Pre-processing*: In order to utilise the clean data for training, it is passed to the load_airbnb(df, labels) which returns a tuples in the format (features, labels) for each example in the dataset.