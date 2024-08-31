from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import PredictionErrorDisplay
from sklearn.utils import shuffle
from millify import prettify
from training import load_model, get_estimator_name

st.set_page_config(layout="wide")

sns.set_context("paper")

@st.cache_data
def get_filepath(directory) -> list[str]:
    file_paths = [file for file in directory.iterdir() if file.is_file()]
    return file_paths

@st.cache_resource
def get_model_dict(filepaths) -> dict:
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
def get_nan_columns(df) -> list[str]:
    nan_cols = [col for col in df.columns if df[col].isna().any()]
    return nan_cols


def get_prediction_vs_actual(_model, X, y):
    X, y = shuffle(X, y)
    fig, ax = plt.subplots()
    scatter_kwgs = {
        "color": "steelblue",
        "s": 15,
        "edgecolor": "steelblue",
        "alpha": 0.6
    }
    line_kwgs = {
        "color": "red"
    }
    _ = PredictionErrorDisplay.from_estimator(
        estimator=_model, 
        X=X, 
        y=y, 
        ax = ax,
        subsample=3000,
        scatter_kwargs=scatter_kwgs,
        kind="actual_vs_predicted",
        line_kwargs=line_kwgs,
        )
    ax.set_title("Prediction vs Actual Values")
    return fig

def get_prediction_vs_residuals(_model, X, y):
    X, y = shuffle(X, y)
    fig, ax = plt.subplots()
    scatter_kwgs = {
        "color": "steelblue",
        "s": 15,
        "edgecolor": "steelblue",
        "alpha": 0.6
    }
    _ = PredictionErrorDisplay.from_estimator(
        estimator=_model, 
        X=X, 
        y=y, 
        ax = ax,
        subsample=3000,
        scatter_kwargs=scatter_kwgs
        )
    ax.set_title("Prediction vs Residuals")
    return fig
    
    

model_directory = Path('./models')
paths_to_model = get_filepath(model_directory)
models_dict = get_model_dict(paths_to_model)
performance_df = pd.read_csv("./models_performance/performance_metrics.csv", index_col=0)

st.title("Diamond Price Prediction :gem:")
st.divider()
tab1, tab2 = st.tabs(["Prediction", "Model Performance"])
with st.sidebar:
    selected_model = st.selectbox("Regression Models",
                 options=models_dict.keys(),
                 key="current_regressor")



with tab1:    
    st.markdown(
        f"<h2 style='font-family: monospace; text-align: center; color: dodgerblue;'>{selected_model}</h2>", 
        unsafe_allow_html=True
        ) 
    
    with st.form("input_features"):
        f_col1, f_col2, f_col3 = st.columns(3, gap='medium')
        with f_col1:
            carat = st.number_input(label="Carat",
                                    min_value=np.nan, 
                                    key="carat",
                                    max_value = 6.0
                                    )
            depth = st.number_input(label="depth",
                                    min_value=np.nan, 
                                    key="depth", 
                                    max_value=100.0
                                    )
            table = st.number_input(label="table",
                                    min_value=np.nan,
                                    key='table',
                                    max_value=100.0
                                    )
        with f_col2:    
            x = st.number_input(label="X",
                                min_value=np.nan,
                                key='x',
                                max_value=12.0
                                )
            y = st.number_input(label="Y",
                                min_value=np.nan, 
                                key='y',
                                max_value=70.0
                                )
            z = st.number_input(label="Z",
                                min_value=np.nan, 
                                key='z',
                                max_value=40.0
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
    
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"<h3 style='font-family: monospace;text-align: center;'>Metrics of {selected_model}</h3>", 
        unsafe_allow_html=True) 
    
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
        
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    X_test = joblib.load("./datasets/X_test")
    y_test = joblib.load("./datasets/y_test")
    
    st.markdown(
        f"<h3 style='font-family: monospace; text-align: center;'>Errors Viz of {selected_model} </h3>", 
        unsafe_allow_html=True)
    
    fcol1, fcol2 = st.columns(2, gap="large")
    
    with fcol1:
        st.pyplot(
            get_prediction_vs_actual(models_dict[selected_model], X_test, y_test)
        )
    
    with fcol2:
        st.pyplot(
            get_prediction_vs_residuals(models_dict[selected_model], X_test, y_test)
        )
    st.divider()