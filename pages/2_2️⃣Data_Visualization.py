import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
import seaborn as sns
from util import chat_gpt, scatter_link_css

with open("tab_icon.png", "rb") as image_file:
    icon_bytes = image_file.read()
st.set_page_config(
    page_title="RegressifyXpert",
    page_icon= icon_bytes,
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://regressifyxpert.github.io/test/index.html',
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
chat_gpt()



if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_deleted' not in st.session_state:
    st.session_state.df_deleted = None
if 'df_changeNA' not in st.session_state:
    st.session_state.df_changeNA = None
if 'df_dropNA' not in st.session_state:
    st.session_state.df_dropNA = None


if st.session_state.df_dropNA is not None:
    df = st.session_state.df_dropNA
else:
    if st.session_state.df_changeNA is not None:
        df = st.session_state.df_changeNA
    else:
        if st.session_state.df_deleted is not None:
            df = st.session_state.df_deleted
        else:
            if st.session_state.df_raw is not None:
                df = st.session_state.df_raw
            else:
                df = None

if "data_convert" not in st.session_state:
    st.session_state.data_convert = df
else:
    if st.session_state.data_convert is None:
        st.session_state.data_convert = df

# if "df_filter" not in st.session_state:
#     st.session_state.df_filter = None

if "scatter_fig" not in st.session_state:
    st.session_state.scatter_fig = None

st.header("Data Visualizing")

if df is not None:
    if st.session_state.missing_value != 0 :
        st.error("Please back to 1️⃣Data Preprocessing page and drop missing values.")
    else:

        # Show data types of each variable
        # data_type1, data_type2= st.columns([0.5,0.5])
        st.subheader("Data Types of Variables:")
        # data_type2.link_button("More Detail....", "https://regressifyxpert.github.io/test/DATA_PREPROCESSING.html#section-2")
        
        data_types = df.dtypes.to_frame().transpose()  # Transpose the DataFrame
        st.dataframe(data_types)

        cat_origin = data_types.columns[(data_types.loc[0] == 'object')|(data_types.loc[0] == 'bool')].tolist()
        num_int_origin = data_types.columns[data_types.loc[0] == 'int64'].tolist()
        num_float_origin = data_types.columns[data_types.loc[0] == 'float64'].tolist()
        num_origin = num_int_origin + num_float_origin



        if 'categorical_vars' not in st.session_state:
            st.session_state.categorical_vars = cat_origin
        if 'numerical_vars' not in st.session_state:
            st.session_state.numerical_vars = num_origin

        
        # Convert categorical variables to numerical variables for visualization
            
        int_convert_to_cat = st.multiselect("Select categorical variables in dataset:", options=num_origin)
            

        if int_convert_to_cat != []:
            if st.button("Convert Categorical Variables"):
                st.session_state.categorical_vars += int_convert_to_cat
                for var in int_convert_to_cat :
                    df[var] = df[var].astype(str)

                st.session_state.data_convert = df
                data_types_trans = df.dtypes.to_frame().transpose()
                st.write("Categorical variables converted to numerical variables for visualization successfully!")
                st.dataframe(data_types_trans)

                numerical_vars = data_types_trans.columns[(data_types_trans.loc[0] == 'int64') | (data_types_trans.loc[0] == 'float')].tolist()
                st.session_state.numerical_vars = numerical_vars
            
                

            

        # Create two columns for visualization
        col1, col2 = st.columns(2)

        # Visualize categorical variables
        with col1:
            st.subheader("Categorical Variables:")
            if len(st.session_state.categorical_vars) == 0:
                st.write("No categorical variables in the dataset.")
            else:
                categorical_var_plot = st.selectbox("Select categorical variable:", options=st.session_state.categorical_vars, key="categorical_var_selectbox")
                with st.expander(f"Show Pie Chart of {categorical_var_plot}"):
                    categorical_var_counts = df[categorical_var_plot].value_counts()
                    fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis objects
                    ax.pie(categorical_var_counts, labels=categorical_var_counts.index, autopct='%1.1f%%')
                    ax.set_title(f"Pie Chart of {categorical_var_plot}")
                    st.pyplot(fig)  # Pass the figure object to st.pyplot()

        # Visualize numerical variables
        with col2:
            st.subheader("Numerical Variables:")
            numerical_var_plot= st.selectbox("Select numerical variable:", options=st.session_state.numerical_vars, key="numerical_var_selectbox")
            with st.expander(f"Show Histogram of {numerical_var_plot}"):
                fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis objects
                ax.hist(df[numerical_var_plot], bins=20)
                ax.set_title(f"Histogram of {numerical_var_plot}")
                ax.set_xlabel(numerical_var_plot)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

        # Create a form for Scatter Plot between Two Numerical Variables
        st.subheader("Scatter Plot between Two Numerical Variables:")
        with st.form(key='scatter_plot_form'):
            col1, col2 = st.columns(2)
            with col1:
                scatter_x = st.selectbox('Select x variable:', st.session_state.numerical_vars, index=0)
            with col2:
                scatter_y = st.selectbox('Select y variable:', st.session_state.numerical_vars, index=len(st.session_state.numerical_vars)-1)
            submitted = st.form_submit_button("Show Scatter Plot")

            # If form is submitted, show the scatter plot
            if submitted:
                fig_scatter = px.scatter(df, x=scatter_x, y=scatter_y, color=None)  # Use scatter_x as color column
                st.plotly_chart(fig_scatter)
                # Calculate and display correlation coefficient
                correlation_coef = pearsonr(df[scatter_x], df[scatter_y])[0]
                st.write(f"Correlation coefficient between {scatter_x} and {scatter_y}: {correlation_coef}")

        # Visualize scatter matrix
        
        st.subheader("Scatter Matrix:")
        scatter_col1, scatter_col2 = st.columns([0.55,0.45])
        scatter_col1.write("Visualizing the relationship between numerical variables")
        scatter_col2.link_button("More Detail....", "https://regressifyxpert.github.io/test/DATA_VISUALIZATION.html")
        scatter_link_css()
        if st.session_state.scatter_fig is not None:
            st.pyplot(st.session_state.scatter_fig)
        else :
            if st.button("Show Scatter Matrix"):
                with st.spinner('Wait for it...'):
                    fig, ax = plt.subplots(figsize=(12, 12))
                    pd.plotting.scatter_matrix(df[st.session_state.numerical_vars], ax=ax)
                    st.session_state.scatter_fig = fig
                    st.pyplot(st.session_state.scatter_fig)

        # Visualize correlation matrix
        st.subheader("Correlation Matrix:")
        scatter_col3, scatter_col4 = st.columns([0.55,0.45])
        scatter_col3.write("Visualizing the correlation between numerical variables")
        scatter_col4.link_button("More Detail....", "https://regressifyxpert.github.io/test/DATA_VISUALIZATION.html#section-7")
        if st.button("Show Correlation Matrix"):
            with st.spinner('Wait for it...'):
                corr_matrix = df[st.session_state.numerical_vars].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                st.pyplot(fig)

      

else:
    st.error("Please upload a CSV file on 1 data preprocessing page.")


pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/1_1️⃣_Data_Preprocessing.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/3_3️⃣Data_Filter.py")


