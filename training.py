import os
import joblib
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer, TransformedTargetRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model._stochastic_gradient import BaseSGD
from sklearn.tree import DecisionTreeRegressor, BaseDecisionTree
from sklearn.ensemble import BaseEnsemble, AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import NeighborsBase
from sklearn.inspection import permutation_importance


def load_dataset(file: str, target= None, return_X_y= False, **kwargs):
    data = pd.read_csv(file, **kwargs)
    df = data.copy()
    if return_X_y:
        X = df.drop([target], axis=1)
        y = df[target]
        return X, y 
    else:
        return df


def build_pipeline(transformer=None, estimator=None, **kwargs) -> Pipeline:
    pipe = make_pipeline(transformer, estimator, **kwargs)
    return pipe


def get_estimator_name(estimator)-> str:
    name = ''
    if isinstance(estimator[-1], RegressorMixin):
        name = estimator[-1].__class__.__name__
    if isinstance(estimator[-1], TransformedTargetRegressor):
        name = ("TransformedTarget_" + estimator[-1].regressor_.__class__.__name__)
    return name


def get_TransformedTargetRegressor(estimator=None):
    quantile = QuantileTransformer(output_distribution='normal')
    ttr = TransformedTargetRegressor(regressor=estimator, transformer=quantile)
    return ttr


def model_performance(fitted_estimator, X_test , y_true) -> pd.DataFrame:
    name = get_estimator_name(fitted_estimator)
    y_pred = fitted_estimator.predict(X_test) 
    metrics_dict = {
        'R-squared': np.round(metrics.r2_score(y_true, y_pred), 2), 
        'Mean Absolute Error': np.round(metrics.mean_absolute_error(y_true, y_pred), 2),
        'Mean Squared Error': np.round(metrics.mean_squared_error(y_true, y_pred), 2),
        'Root Mean Squared Error': np.round(metrics.root_mean_squared_error(y_true, y_pred), 2)
        }
    df_scores = pd.DataFrame(metrics_dict, index=[name])
    return df_scores


def get_feature_importance(name, fitted_pipeline):
    estimator = fitted_pipeline[-1]
    feature_names = fitted_pipeline[:-1].get_feature_names_out()
    
    if isinstance(estimator, TransformedTargetRegressor):
        coeffiecients = estimator.regressor_.coef_
    if isinstance(estimator, LinearModel):
        coeffiecients = estimator.coef_
    if isinstance(estimator, BaseSGD):
        coeffiecients = estimator.coef_
    if isinstance(estimator, BaseDecisionTree):
        coeffiecients = estimator.feature_importances_
    if isinstance(estimator, BaseEnsemble):
        coeffiecients = estimator.feature_importances_
    # if isinstance(estimator, NeighborsBase):
    #     per = permutation_importance(estimator, )
    features_df = pd.DataFrame(data=coeffiecients, index=feature_names, columns=[name])
    return features_df


def get_knn_feature_importance(knn, X, y, **kwargs):
    permutation_result = permutation_importance(knn, X, y)
    feature_imp = permutation_result.importances_mean
    return feature_imp


def plot_feature_importance(df):
    col_name = df.columns.tolist()[0]
    sorted_df = df.sort_values(col_name)
    fig, ax = plt.subplots(figsize=(6,8))
    sorted_df.plot.barh(ax=ax, xlabel='Coefficients', title="Feature Importance in " + col_name, legend=False)
    ax.axvline(linestyle='--', color='k', linewidth=1)
    return fig


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

    num_selector = make_column_selector(dtype_include='number')
    cat_selector = make_column_selector(dtype_exclude='number')

    preprocessor = make_column_transformer(
    (StandardScaler(), num_selector),
    (OneHotEncoder(), cat_selector)
    )

    model_fitted = {}
    model_scores = {}
    model_results = []
    saving_directory = "./models"
    
    regressors = [
        LinearRegression(),
        Ridge(),
        SGDRegressor(),
        Lasso(max_iter=3000),
        ElasticNet(),
        LinearSVR(max_iter=5000),
        DecisionTreeRegressor(), 
        AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=100),
        RandomForestRegressor(),
        ExtraTreesRegressor(),
        KNeighborsRegressor()
    ]
    
    
    for reg in regressors:
        model = build_pipeline(preprocessor, reg)
        model.fit(X_train, y_train)
        model_fitted.update({get_estimator_name(model): model})
        model_scores.update({get_estimator_name(model): model.score(X_test, y_test)})
        model_results.append(model_performance(model, X_test, y_test))
        save_model(model, saving_directory)
        
    """TransformedTargetRegressor
    """    
    for reg in regressors[:3]:
        ttr_reg = get_TransformedTargetRegressor(reg)
        ttr_model = build_pipeline(preprocessor, ttr_reg)
        ttr_model.fit(X_train, y_train)
        model_fitted.update({get_estimator_name(ttr_model): ttr_model})
        model_scores.update({get_estimator_name(ttr_model): ttr_model.score(X_test, y_test)})
        model_results.append(model_performance(ttr_model, X_test, y_test))
        save_model(ttr_model, saving_directory)
        
    results_df = pd.concat(model_results, axis=0)
    results_df.to_csv('./models_performance/performance_metrics.csv')
    pprint(model_scores)
