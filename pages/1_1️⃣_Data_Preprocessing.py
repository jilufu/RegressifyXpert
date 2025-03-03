import streamlit as st
import pandas as pd
import numpy as np
from util import chat_gpt, store_value, load_value, data_preprocessing_page_css

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

st.header("Data Preprocessing")
st.subheader("Upload a CSV file:")
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'nrows_raw' not in st.session_state:
    st.session_state.nrows_raw = None
if 'df_deleted' not in st.session_state:
    st.session_state.df_deleted = None
if 'df_changeNA' not in st.session_state:
    st.session_state.df_changeNA = None
if 'df_dropNA' not in st.session_state:
    st.session_state.df_dropNA = None

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if all(df[col].dtype == 'object' for col in df.columns):
        st.error("Please upload a dataset containing numeric variables.")
    else:
        st.session_state.df_raw = df
        st.session_state.nrows_raw = df.shape[0]

df = st.session_state.df_raw

# Retrieve the DataFrame from session state

if df is not None:
        # Show dataframe
        st.write(f"Preview of the uploaded dataset: `{df.shape[0]}`the numbers of rows ")
        st.dataframe(df)

        # Allow user to delete rows and columns
        st.subheader("Delete rows and columns:")

        # Create two columns for options
        col_delete_rows, col_delete_columns = st.columns(2)

        if 'delete_rows' not in st.session_state:
            st.session_state.delete_rows = []

        if 'delete_cols' not in st.session_state:
            st.session_state.delete_cols = []

        # Left column: Delete rows
        with col_delete_rows:
            st.write("Delete rows:")
            load_value("delete_rows")
            st.multiselect("Select rows to delete:", options=df.index.tolist(), key="_delete_rows", on_change=store_value, args=["delete_rows"])

        # Right column: Delete columns
        with col_delete_columns:
            st.write("Delete columns:")
            load_value("delete_cols")
            st.multiselect("Select columns to delete:", options=df.columns.tolist(), key="_delete_cols", on_change=store_value, args=["delete_cols"])

        left1, middle , right1 = st.columns([0.3,0.4,0.3])
        with middle:
            if st.button("Delete Selected Rows and Columns"):
                if len(st.session_state.delete_rows)>0 or len(st.session_state.delete_cols)>0:
                    df = df.drop(index=st.session_state.delete_rows, columns=st.session_state.delete_cols)
                    st.session_state.df_deleted = df
                    
                else:
                    st.error("Please select rows or columns to delete.")

        if st.session_state.df_deleted is not None:
            st.write(":green[deleted successfully!] There is filtered data: the sample size is now", st.session_state.df_deleted.shape[0])
            st.dataframe(st.session_state.df_deleted)
            df = st.session_state.df_deleted        
        else:
            df = st.session_state.df_raw

        if 'missing_value'  not in st.session_state:
            st.session_state.missing_value = None
        # Show missing values information
        missing_values = df.isna().sum()
        missing_values_transposed = missing_values.to_frame().T  # Transpose the DataFrame
        missing_col1, missing_col2 = st.columns([0.50,0.5])
        missing_col1.subheader("Missing values information:")
        missing_col2.link_button("More Detail....", "https://regressifyxpert.github.io/test/DATA_PREPROCESSING.html#section-1")
        data_preprocessing_page_css()

        st.dataframe(missing_values_transposed)
        if missing_values.sum() == 0:
            st.write(":green[No missing values found in the dataset.]")
        st.session_state.missing_value = missing_values.sum()

        # Additional functionality for handling missing values
        st.subheader("Handle potential missing values:")
        st.write("If missing values are represented by values other than NA, please enter the following information: In the following variables,")

        if 'var_missing_values' not in st.session_state:
            st.session_state.var_missing_values = None

        if 'missing_values_representation' not in st.session_state:
            st.session_state.missing_values_representation = '...'

        # Allow user to select variables with potential missing values
        st.session_state.var_missing_values = st.multiselect("Select variables with potential missing values:", options=df.columns.tolist(),default=st.session_state.var_missing_values)

        # Display text input for missing value representation
        st.session_state.missing_values_representation = st.text_input("Enter missing value representation:", value=st.session_state.missing_values_representation)

        # Show missing value representation
        st.write(f"Missing values are represented as'{st.session_state.missing_values_representation}'")
    

        # Update missing value representation in DataFrame
        if st.button("Update Missing Value Representation"):
            if st.session_state.var_missing_values is not None and st.session_state.missing_values_representation != '...':
                try:
                    missing_value_representation = float(st.session_state.missing_values_representation)
                except ValueError:
                    # 如果無法轉換為整數，則保留為字串型態
                    missing_value_representation = st.session_state.missing_values_representation
                for variable in st.session_state.var_missing_values:
                    # df.loc[:, variable].replace(missing_value_representation, np.nan, inplace=True)
                    df.loc[:, variable] = df.loc[:, variable].replace(missing_value_representation, np.nan)
                
                st.write("Missing value representation updated successfully!")
                st.session_state.df_changeNA = df
                
            else:
                st.warning("Please select variables with potential missing values.")

        # Show missing values information again after updates
        if st.session_state.df_changeNA is not None:
            missing_values = st.session_state.df_changeNA.isna().sum()
            missing_values_transposed = missing_values.to_frame().T
            st.caption("Missing values information after transformation:")
            st.dataframe(missing_values_transposed)
            #st.dataframe(st.session_state.df_changeNA)
            st.session_state.missing_value = missing_values.sum()

        if st.session_state.missing_value != 0:
        
            st.subheader("Delete rows with missing values")

            if st.session_state.df_changeNA is not None:
                df = st.session_state.df_changeNA
            else:
                if st.session_state.df_deleted is not None:
                    df = st.session_state.df_deleted
                else:
                    df = st.session_state.df_raw


            if st.button("Delete missing values"):
                df.dropna(inplace=True)
                st.write(f"Successfully deleted all rows with missing values！ The number of rows in the dataset is now: {df.shape[0]}")
                st.dataframe(df)
                st.session_state.df_dropNA = df
                st.session_state.missing_value = 0


else:
    st.error("Please upload a CSV file.")
    @st.cache_data
    def dowmload():
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        dataset = pd.read_csv('data.csv')
        return dataset.to_csv().encode("utf-8")

    csv = dowmload()
    demo1, demo2 = st.columns([0.25,0.75])
    with demo1:
        st.write("Data File Demo Format")
    with demo2:
        st.download_button(
            label="Download Demo Data",
            data=csv,
            file_name="exampleData.csv",
            mime="text/csv",
        )
    
    st.image("data_example.jpg", use_column_width=True)






pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("🏠Main_Page.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/2_2️⃣Data_Visualization.py")







