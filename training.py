import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer, TransformedTargetRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error, root_mean_squared_error
)


def load_dataset(file: str, target= None, return_X_y= False, **kwargs) -> pd.DataFrame | tuple:
    data = pd.read_csv(file, **kwargs)
    df = data.copy()
    if return_X_y:
        X = df.drop([target], axis=1)
        y = df[target]
        return X, y 
    else:
        return df


def get_estimator_name(estimator)-> str:
    name: str = ''
    if isinstance(estimator[-1], RegressorMixin):
        name = estimator[-1].__class__.__name__
    if isinstance(estimator[-1], TransformedTargetRegressor):
        name = ("TransformedTarget_" + estimator[-1].regressor_.__class__.__name__)
    return name


def get_TransformedTargetRegressor(estimator=None, **kwargs):
    quantile = QuantileTransformer(output_distribution='normal')
    ttr = TransformedTargetRegressor(regressor=estimator, transformer=quantile, **kwargs)
    return ttr


def model_performance(fitted_estimator, X_test , y_true) -> dict[str, np.ndarray[np.float64]]:
    y_pred: np.ndarray = fitted_estimator.predict(X_test) 
    metrics_dict: dict[str, np.ndarray[np.float64]] = {
        'R-squared': np.round(r2_score(y_true, y_pred), 2), 
        'Mean Absolute Error': np.round(mean_absolute_error(y_true, y_pred), 2),
        'Mean Squared Error': np.round(mean_squared_error(y_true, y_pred), 2),
        'Root Mean Squared Error': np.round(root_mean_squared_error(y_true, y_pred), 2)
        }
    return metrics_dict


def save_model(model, filepath= None, **kwargs) -> None:
    if filepath: 
        model_name = os.path.join(filepath, get_estimator_name(model))
    else:
        model_name = get_estimator_name(model) 
    joblib.dump(model, model_name, **kwargs)

  
def load_model(file, **kwargs):
    model = joblib.load(file, **kwargs)
    return model


if __name__ == '__main__':
    data = load_dataset('./datasets/cleaned_data.csv', index_col=0)
    
    print(data.head())
    
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    joblib.dump(X_train, "./datasets/X_train")
    joblib.dump(X_test, "./datasets/X_test")
    joblib.dump(y_train, "./datasets/y_train")
    joblib.dump(y_test, "./datasets/y_test")
    
    numerical_selector = make_column_selector(dtype_include='number')
    categorical_selector = make_column_selector(dtype_exclude='number')

    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_selector),
        (OneHotEncoder(), categorical_selector)
    )
    
    model_names = []
    model_results = []
    results_with_df = []
    saving_directory = "./models"
    
    regressors = [
        RandomForestRegressor(n_estimators=50),
        KNeighborsRegressor(weights="distance")
    ]
    
    feature_selector = SequentialFeatureSelector(
        estimator= DecisionTreeRegressor(max_depth=20)
    )
    
    for reg in regressors:
        model = make_pipeline(preprocessor, feature_selector,reg)
        model.fit(X_train, y_train)
        model_results.append(model_performance(model, X_test, y_test))
        model_names.append(get_estimator_name(model))
        save_model(model, saving_directory)
        
    ridge = make_pipeline(
        preprocessor, get_TransformedTargetRegressor(Ridge())
    )
    ridge.fit(X_train, y_train)
    model_results.append(model_performance(ridge, X_test, y_test))
    model_names.append(get_estimator_name(ridge))
    save_model(ridge, saving_directory)

    for name, result in zip(model_names, model_results):
        results_with_df.append(pd.DataFrame(result, index=[name]))
        
    results_df = pd.concat(results_with_df, axis=0)
    results_df.to_csv('./models_performance/performance_metrics.csv')
    
    globals().clear()