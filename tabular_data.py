import pandas as pd

class DataCleaning():

    def remove_rows_with_missing_ratings(self, dataframe):
        '''Removes the rows with missing values for any of the ratings columns.'''

        columns = ["Cleanliness_rating", "Accuracy_rating", 
                   "Communication_rating", "Location_rating", 
                   "Check-in_rating", "Value_rating"]
        
        for c in columns:
            df_missing = dataframe[dataframe[c].isnull()]
            dataframe.drop(df_missing.index, inplace= True)

        dataframe.drop("Unnamed: 19", axis= 1, inplace= True)

        return dataframe


    def combine_description_strings(self, dataframe):
        '''Combines the lists of strings in the description column into sings string'''

        #Dropping any rows with missing descriptions
        df_missing_description = dataframe[dataframe["Description"].isna() == True]
        dataframe.drop(df_missing_description.index, inplace= True)

        #Cleaning descriptions and joining them into one string from list
        dataframe["Description"] = dataframe["Description"].str.replace("'", "")
        dataframe["Description"] = dataframe["Description"].str.replace('"', "")
        dataframe["Description"] = dataframe["Description"].str.replace("About this space,", "")
        dataframe["Description"] = dataframe["Description"].str.join("")

        return dataframe
    

    def set_default_feature_value(self, dataframe):
        '''Replaces NaN in guest, beds, bathrooms and bedrooms columns with 1s'''

        df_fillna = dataframe[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1)

        df = dataframe
        df[["guests", "beds", "bathrooms", "bedrooms"]] = df_fillna

        return df
    

def clean_tabular_data(dataframe):
    cleaner = DataCleaning()
    df = cleaner.remove_rows_with_missing_ratings(dataframe)
    df = cleaner.combine_description_strings(df)
    df = cleaner.set_default_feature_value(df)
    return df

def load_airbnb(dataframe, label):
    '''Removes columns containing text and returns the features and
    selected label column as tuples of arrays (features, label)'''

    #Drop all columns that contain text data
    features_without_labels = dataframe.drop(label, axis= 1)

    text_features = features_without_labels.select_dtypes(include=["object"])
    features_df = features_without_labels.drop(text_features, axis= 1)
    features_df = features_df.drop("Unnamed: 0", axis= 1)

    #Assign features and labels to be returned as tuples
    labels = dataframe[label].to_numpy()
    features = features_df.to_numpy()

    return (features, labels)


if __name__ == "__main__":

    data = pd.read_csv("./airbnb-property-listings/tabular_data/listing.csv")

    data = clean_tabular_data(data)

    data.to_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")

    print(data.isna().sum())


