from tabular_data import load_airbnb
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def custom_tune_regression_model_hyperparameters(model, training_set, validation_set, test_set, param_dict):
    #TODO code function

    return best_model, best_param_dict, perfomance_dict

if __name__ == "__main__":

    data = pd.read_csv("./airbnb-property-listings/tabular_data/clean_tabular_data.csv")

    X, y = load_airbnb(data, "Price_Night")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

    reg_model = SGDRegressor()

    reg_model.fit(X_train, y_train)

    y_pred_train = reg_model.predict(X_train)
    y_pred = reg_model.predict(X_test)

    #Calculating the RMSE and R^2 value for the training set
    rmse_train = mean_squared_error(y_train, y_pred_train, squared= False)
    r2_train = r2_score(y_train, y_pred_train)

    #Calculating the RMSE and R^2 values for the test set
    rmse_test = mean_squared_error(y_test, y_pred, squared= False)
    r2_test = r2_score(y_test, y_pred)



    print("The training RMSE of the model is ", rmse_train)
    print("The test RMSE of hte model is ", rmse_test)

    print("The R^2 value for the training set is ", r2_train)
    print("The R^2 value for the test set is ", r2_test)



