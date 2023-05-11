import pandas as pd

class DataCleaning():

    def remove_rows_with_missing_ratings(self, dataframe):
        '''Removes the rows with missing values for any of the ratings columns.'''

        df_cleanliness_missing = dataframe[dataframe["Cleanliness_rating"].isnull()]
        dataframe.drop(df_cleanliness_missing.index, inplace= True)

        df_accuracy_missing = dataframe[dataframe["Accuracy_rating"].isnull()]
        dataframe.drop(df_accuracy_missing.index, inplace= True)

        df_communication_missing = dataframe[dataframe["Communication_rating"].isnull()]
        dataframe.drop(df_communication_missing.index, inplace= True)

        df_location_missing = dataframe[dataframe["Location_rating"].isnull()]
        dataframe.drop(df_location_missing.index, inplace= True)

        df_check_in_missing = dataframe[dataframe["Check-in_rating"].isnull()]
        dataframe.drop(df_check_in_missing.index, inplace= True)

        df_value_missing = dataframe[dataframe["Value_rating"].isnull()]
        dataframe.drop(df_value_missing.index, inplace= True)

        dataframe.drop("Unnamed: 19", axis= 1, inplace= True)

        return dataframe


    def combine_description_strings(self, dataframe):
        #TODO - combine lists of strings in Description column (don't implement a from-scratch solution to parse the string into list)
        #Dropping any rows with missing descriptions
        df_missing_description = dataframe[dataframe["Description"].isna() == True]
        dataframe.drop(df_missing_description.index, inplace= True)

        #Cleaning descriptions and joining them into one string from list
        dataframe["Description"] = dataframe["Description"].str.replace("'", "")
        dataframe["Description"] = dataframe["Description"].str.replace('"', "")
        dataframe["Description"] = dataframe["Description"].str.replace("About this space,", "")
        dataframe["Description"] = dataframe["Description"].str.join("")
        dataframe["Description"] = dataframe["Description"].astype("str")

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

    data = clean_tabular_data(data)

    print(data.info())

    # data.to_csv("clean_listing.csv")

