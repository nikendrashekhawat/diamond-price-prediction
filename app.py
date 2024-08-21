import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.neighbors._base import NeighborsBase
from millify import prettify
from training import get_feature_importance, plot_feature_importance
from training import load_model, get_estimator_name

st.set_page_config(layout="wide")

sns.set_theme(
    context="paper", 
    style='dark', 
    rc = {
        'figure.facecolor' : '#EAEAF2',
        'axes.edgecolor' : '0.25'
        }
    )

@st.cache_data
def get_filepath(directory):
    file_paths = [file for file in directory.iterdir() if file.is_file()]
    return file_paths

@st.cache_resource
def get_model_dict(filepaths):
    models = {}
    for file in filepaths:
        m = load_model(file)
        models.update({get_estimator_name(m) : m})
    return models

@st.cache_data
def check_nan(df) -> bool:
    boolean = df.isna().values.any(axis=1)
    return boolean

@st.cache_data
def get_nan_columns(df) -> list:
    nan_cols = [col for col in df.columns if df[col].isna().any()]
    return nan_cols


model_directory = Path('./models')
paths_to_model = get_filepath(model_directory)
models_dict = get_model_dict(paths_to_model)
performance_df = pd.read_csv("./models_performance/performance_metrics.csv", index_col=0)

st.title("Diamond Price Prediction :gem:")
st.divider()
tab1, tab2 = st.tabs(["Model Prediction", "Model Performance"])
with st.sidebar:
    selected_model = st.selectbox("Regression Models",
                 options=models_dict.keys(),
                 key="current_regressor")



with tab1:    
    st.subheader(f"Model :blue[{selected_model}] is selected for prediction") 
    
    with st.form("input_features"):
        f_col1, f_col2, f_col3 = st.columns(3, gap='medium')
        with f_col1:
            carat = st.number_input( label="Carat",
                                    min_value=np.nan, 
                                    key="carat"
                                    )
            depth = st.number_input(label="depth",
                                    min_value=np.nan, 
                                    key="depth"
                                    )
            table = st.number_input(label="table",
                                    min_value=np.nan,
                                    key='table'
                                    )
        with f_col2:    
            x = st.number_input(label="X",
                                min_value=np.nan,
                                key='x'
                                )
            y = st.number_input(label="Y",
                                min_value=np.nan, 
                                key='y',
                                )
            z = st.number_input(label="Z",
                                min_value=np.nan, 
                                key='z'
                                )
        with f_col3: 
            cut = st.selectbox(label="Cut Quality", 
                                options=['Fair', 'Very Good', 'Good', 'Premium', 'Ideal'],
                                index=None, 
                                key='cut',
                                help="Cut Quality [Fair, Good, Very Good, Premium, Ideal(best)]")
            color = st.selectbox(label="Color", 
                                options=['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                index=None,
                                key='color', 
                                help="Diamond Color:J(worst) to D(best)")
            clarity = st.selectbox(label="Clarity",
                                    options=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                                    index=None, 
                                    key='clarity', 
                                    help="How clear the diamond is [I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)]")
        submit_button = st.form_submit_button("Predict")
        
    features_dict = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }

    input_df = pd.DataFrame(features_dict, index=[0])

    if submit_button:
        if check_nan(input_df):
            nan_cols_ls = get_nan_columns(input_df)
            nan_cols_str = ', '.join(nan_cols_ls)
            st.error("InputValueError!", icon=":material/error:")
            st.error(f"Input feature {nan_cols_str} contains unsupported/invalid value.", icon=":material/exclamation:")
            st.warning("Prediction model expects valid numeric/categorical value.", icon=":material/data_alert:")
        else:
            prediction = np.round(models_dict[selected_model].predict(input_df), 2)
            prediction_str = prediction.astype(np.str_)
            p_col1, p_col2, p_col3 = st.columns(3, gap="large")
            
            with p_col1.container(height=200, border=False):
                st.header("Estimated Price:")
                
            with p_col2.container(height=100, border=False):
                st.header(f"${prettify(prediction_str[0])}")
    
with tab2:
    st.subheader(f"Metrics of :blue[{selected_model}]:")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4, gap="medium")
    with mcol1.container(height=150):
        st.subheader("R-squared")
        st.subheader(f":gray-background[{performance_df.loc[selected_model, "R-squared"]}]")
    with mcol2.container(height=150):
        st.subheader("MAE")
        st.subheader(f":gray-background[{performance_df.loc[selected_model, "Mean Absolute Error"]}]")
    with mcol3.container(height=150):
        st.subheader("MSE")
        st.subheader(f":gray-background[{performance_df.loc[selected_model, "Mean Squared Error"]}]")
    with mcol4.container(height=150):
        st.subheader("RMSE")
        st.subheader(f":gray-background[{performance_df.loc[selected_model, "Root Mean Squared Error"]}]")
    
    st.divider()
    st.subheader("Feature Importance")
    
    if not isinstance(models_dict[selected_model][-1], NeighborsBase): 
        fig_feature_imp = (get_feature_importance(selected_model, models_dict[selected_model])
                    .pipe(plot_feature_importance)
        )
        with st.container(border=True):
            st.pyplot(fig_feature_imp)
        # st.bar_chart(data=f_imp, horizontal=True, width=800, use_container_width=False)
