import pandas as pd

class DataCleaning():

    def remove_rows_with_missing_ratings(self, dataframe):
        #TODO - remove rows with missing values in ratings columns
        print("Removing empty rows.")
        return dataframe

    def combine_description_strings(self, dataframe):
        #TODO - combine lists of strings in Description column (don't implement a from-scratch solution to parse the string into list)
        print("Combining strings.")
        return dataframe

    def set_default_feature_value(self, dataframe):
        #TODO - replace missing data entries in "guests", "beds", "bathrooms" and "bedrooms" columns with 1s
        print("Setting default values.")
        return dataframe

def clean_tabular_data(dataframe):
    cleaner = DataCleaning()
    df = cleaner.remove_rows_with_missing_ratings(dataframe)
    df = cleaner.combine_description_strings(df)
    df = cleaner.set_default_feature_value(df)
    return df

if __name__ == "__main__":

    data = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")

    print(data.describe())

    data = clean_tabular_data(data)

    data.to_csv("clean_listing.csv")

