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

        dataframe[["guests", "beds", "bathrooms", "bedrooms"]].fillna(1, inplace= True)

        return dataframe
    

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
    text_columns = dataframe.select_dtypes(include=["object"])
    dataframe.drop(text_columns, axis= 1, inplace= True)

    dataframe.drop("Unnamed: 0", axis= 1, inplace= True)

    #Assign features and labels to be returned as tuples
    labels = dataframe[label].to_numpy()
    features = dataframe.drop(label, axis= 1).to_numpy()

    tuples = []
    
    for idx in range(len(labels)):

        tuples.append((features[idx], labels[idx]))

    return tuples


if __name__ == "__main__":

    data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")

    data = load_airbnb(data, "Price_Night")

    print(data)


