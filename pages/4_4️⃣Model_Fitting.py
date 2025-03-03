import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import PredictionErrorDisplay
from util import chat_gpt, store_value, load_value, scatter_explain, handling_dummy, user_choose_model_vars,data_preprocessing_page_css

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



if "df_filter" not in st.session_state:
    st.session_state.df_filter = None
if "data_convert" not in st.session_state:
    st.session_state.data_convert = None


if st.session_state.df_filter is None:
    df = st.session_state.data_convert

elif st.session_state.df_filter.shape[0]==0:
    df = None
    error_text = "the dataset is empty. Please back to data filter page"
else:
    df = st.session_state.df_filter


# title
st.header("Model Fitting")

if "user_choose_y" not in st.session_state:
    st.session_state.user_choose_y = None
if "user_choose_x_num" not in st.session_state:
    st.session_state.user_choose_x_num = []
if "user_choose_x_cat" not in st.session_state:
    st.session_state.user_choose_x_cat = []

if "dummy_varName" not in st.session_state:
    st.session_state.dummy_varName = []

if "model_dataset" not in st.session_state:
    st.session_state.model_dataset = None

if "est_function" not in st.session_state:
    st.session_state.est_function = ""

if "mean_est_function" not in st.session_state:
    st.session_state.mean_est_function = ""

if "ols_function_interpre" not in st.session_state:
    st.session_state.ols_function_interpre = None

if "final_data" not in st.session_state:
    st.session_state.final_data = None




if df is not None:

    df = df.reset_index(drop=True)
    
    categorical_vars = st.session_state.categorical_vars
    numerical_vars = st.session_state.numerical_vars

    for var in categorical_vars:
        df[var] = df[var].str.replace('_', ' ')

    model_y, model_x = user_choose_model_vars(numerical_vars, categorical_vars)
    
    if model_y is not None:
        st.session_state.user_choose_y  =  model_y   
     
    if len(model_x) > 0:
        # show the categorical variables, numerical variables
        numeric_x = list(set(model_x) & set(numerical_vars))
        category_x = list(set(model_x) & set(categorical_vars))
        st.session_state.user_choose_x_num = numeric_x
        st.session_state.user_choose_x_cat = category_x

    if len(st.session_state.user_choose_x_num)>0:
        text_value_num = ", ".join(st.session_state.user_choose_x_num)
    else:
        text_value_num = None
    
    if len(st.session_state.user_choose_x_cat)>0:
        text_value_cat = ", ".join(st.session_state.user_choose_x_cat)
    else:
        text_value_cat = None

    container_ModelFitting12 = st.container(border=True)
    with container_ModelFitting12:
        st.write("<div style='padding-bottom: 0.5rem;'>Selected Variable Categories：</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_area( label="Y-Numerical Variable",value= st.session_state.user_choose_y)
        with col2:
            st.text_area(label="X-Numerical Variables",value=text_value_num)
        with col3:
            st.text_area(label="X-Categorical Variables",value=text_value_cat)

    
    # 確認變數間的關係
    # drawing scatter matrix plot for all selected numeric variables \
        st.write("<div style='padding-bottom: 0.5rem;'>Plot Scatterplot Matrix of Selected Continuous Variables：</div>", unsafe_allow_html=True)
        
        if st.session_state.user_choose_y is not None: #scatter dummy model_data

            st.session_state.dummy_varName = []
            if len(st.session_state.user_choose_x_cat)>0 and len(st.session_state.user_choose_x_num)>0:
                df_numeric = scatter_explain(df)
                df_dummy, need_to_dummy = handling_dummy(df)
                if df_dummy is not None:
                    model_data = pd.concat([df_numeric, df_dummy], axis=1)
                else:
                    model_data = df_numeric

                st.session_state.model_dataset = model_data


            elif len(st.session_state.user_choose_x_num)>0:
                df_numeric = scatter_explain(df)
                model_data = df_numeric
                # # for categorical variables, draw boxplot for each variable and scatter plot for x is order and group by the categorical variables
                # # 繪製箱形圖和散點圖 
                st.session_state.model_dataset = model_data

            elif len(st.session_state.user_choose_x_cat)>0:
                st.info("There is no numeric independent variable.")
                df_dummy, need_to_dummy = handling_dummy(df)
                if df_dummy is not None:
                    df_y = df.loc[:,st.session_state.user_choose_y]
                    model_data = pd.concat([df_y, df_dummy], axis=1)
                    st.session_state.model_dataset = model_data                
                else:
                    st.error("Please select more independent variables to fit the model.") 
                    st.session_state.model_dataset = None 
                
            else:
                st.info("Please select independent variables.")
        
        
        else:
            st.info("Please select variables.")
            
    # build the model  
    if st.session_state.model_dataset is not None:
        css1, css2 = st.columns([0.3, 0.7])
        css1.subheader("Model Selection")
        css2.link_button("More Detail....", "https://regressifyxpert.github.io/test/MODEL_FITTING.html#section-13")
        data_preprocessing_page_css()
        # var name
        ynam = model_data.columns[0]
        NumName = []
        if st.session_state.dummy_varName != []:
            dummy_flattened_list = st.session_state.dummy_varName
            non_bool_columns = model_data.select_dtypes(exclude='bool').columns.tolist()
            NumName = non_bool_columns[1:]
            xnam = NumName + dummy_flattened_list
            x_square = [f"{var}^2" for var in NumName]
            x_log = [f"log({var})" for var in NumName]
            

        else:
            xnam = model_data.columns[1:].to_list()
            NumName = xnam
            x_square = [f"{var}^2" for var in xnam]
            x_log = [f"log({var})" for var in xnam]
            
         
        x_exp = [f"exp({var})" for var in xnam]
        x_interaction = [] 
        if len(xnam) > 1:
            for i in range(len(xnam)):
                for j in range(i+1, len(xnam)):
                    x_interaction.append(f"{xnam[i]}*{xnam[j]}")


        # choose the model form
        Model_selection_1 = st.container(border=False) 
        with Model_selection_1 :
            yform_container, xform_container_0, button_all = st.columns([0.3,0.55, 0.15])
            yform_option = [ynam, f"log({ynam})", f"{ynam}^2"]
            if "boxcox_y" not in st.session_state:
                st.session_state.boxcox_y = None
            if st.session_state.boxcox_y is not None:
                yform_option.append(f"boxcox({ynam})")
            
            if "y_form" not in st.session_state:
                st.session_state.y_form = ynam

            load_value("y_form")
            yform_container.radio("Select the dependent variable form", options=yform_option, key="_y_form", on_change=store_value, args=["y_form"])
            


            if "x_first_order_form" not in st.session_state:
                st.session_state.x_first_order_form = []
            else:
                st.session_state.x_first_order_form = [
                        var for var in st.session_state.x_first_order_form if var in xnam
                    ]
                                
            if button_all.button("Select All"):
                st.session_state.x_first_order_form = xnam

            load_value("x_first_order_form")
            xform_container_0.multiselect("Select independent variables for the first-order form", options=xnam, key="_x_first_order_form", on_change=store_value, args=["x_first_order_form"])
            
        Model_selection_2 = st.container(border=False) 
        with Model_selection_2 :
            xform_container_1 ,xform_container_2 =st.columns(2)
            if "x_second_order_form" not in st.session_state:
                st.session_state.x_second_order_form = []
            #use multiselect to select the independent variables of the second-order form
            load_value("x_second_order_form")
            xform_container_1.multiselect("Select independent variables for the second-order form", options=x_square, key="_x_second_order_form",on_change=store_value, args=["x_second_order_form"])

            

            if "x_interaction_form" not in st.session_state:
                st.session_state.x_interaction_form = []
            #use multiselect to select the independent variables of the interaction form
            load_value("x_interaction_form")
            xform_container_2.multiselect("Select independent variables for the interaction form", options=x_interaction, key="_x_interaction_form",on_change=store_value, args=["x_interaction_form"])

            xform_container_3 ,xform_container_4 =st.columns(2)
            if "x_log_form" not in st.session_state:
                st.session_state.x_log_form = []
            load_value("x_log_form")
            #use multiselect to select the independent variables of the nature log form
            xform_container_3.multiselect("Select independent variables for the log form", options=x_log, key="_x_log_form",on_change=store_value, args=["x_log_form"])

            if "x_exp_form" not in st.session_state:
                st.session_state.x_exp_form = []
            load_value("x_exp_form")
            #use multiselect to select the independent variables of the exp form
            xform_container_4.multiselect("Select independent variables for the exp form", options=x_exp, key="_x_exp_form",on_change=store_value, args=["x_exp_form"])

        
        Model_selection_3 = st.container(border=False) 
        with Model_selection_3 :
            xcusform_1 ,xcusform_2 =st.columns([0.4,0.6])
            #use multiselect to select the independent variables of the custom form
            if "x_custom_only_var" not in st.session_state:
                st.session_state.x_custom_only_var = []
            load_value("x_custom_only_var")
            xcusform_1.multiselect("Select independent variables for the custom form", options=NumName,key="_x_custom_only_var", on_change=store_value, args=["x_custom_only_var"])
                
            if "x_custom_order" not in st.session_state:
                st.session_state.x_custom_order = -1.0
            load_value("x_custom_order")
            xcusform_1.number_input("Please input the order of the custom form",min_value=-3.0, max_value=3.0, step=0.1, key="_x_custom_order", on_change=store_value, args=["x_custom_order"])
                
            x_custom_order = round(st.session_state.x_custom_order, 1)
            x_custom_option = [f"{var}^{{{x_custom_order}}}" for var in st.session_state.x_custom_only_var]
            
            if "x_custom_form" not in st.session_state:
                st.session_state.x_custom_form = []
            # load_value("x_custom_form")
            st.session_state.x_custom_form = xcusform_2.multiselect("Select independent variables for the custom form", options=x_custom_option, key="_x_custom_form", default=x_custom_option)
                
            all_x_form = st.session_state.x_first_order_form + st.session_state.x_second_order_form + st.session_state.x_interaction_form + st.session_state.x_log_form + st.session_state.x_exp_form + st.session_state.x_custom_form
            
        
        if len(all_x_form) > 0:
            #st.subheader("Model Form")
            st.write("The multiple regression equation with an intercept term can be written as:")
            equation_tab = f"$${st.session_state.y_form} = β₀ + "
            for idx, var in enumerate(all_x_form, start=1):
                equation_tab += f"β_{{{idx}}} {var} + "
            equation_tab += f"ε $$"
            st.markdown(equation_tab)
            markdown_text = """
             **Assumptions of the error term $\\varepsilon $:**
            1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
            2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
            3. The error term $ \\varepsilon $ is normally distributed.
            4. The error terms are independent of each other.
            """
            st.markdown(markdown_text)

        
        
            # data for model fitting
            fit_container1, fit_container2 = st.columns([0.58,0.42])
            fit_container1.subheader("Use OLS Method to fit the model :")
            
            if fit_container2.button("Run Model"):
                if "boxcox_lambda" not in st.session_state:
                    st.session_state.boxcox_lambda = None
                else:
                    st.session_state.boxcox_lambda = None

                y_index = yform_option.index(st.session_state.y_form)

                if y_index == 0 :
                    y_data = model_data.iloc[:, 0]
                    
                elif y_index == 1 :
                    y_data = np.log(model_data.iloc[:, 0])
                    
                elif y_index == 2 :
                    y_data = model_data.iloc[:, 0]
                    y_data = y_data**2
                    
                else :
                    y_data = pd.Series(st.session_state.boxcox_y, name=st.session_state.y_form)
                
                y_data.name = st.session_state.y_form
                bool_columns = model_data.select_dtypes(include=bool).columns
                model_data[bool_columns] = model_data[bool_columns].astype(int)
                model_data_to_trans = model_data.copy()

                if "prediction_var" not in st.session_state:
                    st.session_state.prediction_var = []
                else:
                    st.session_state.prediction_var = []
                
                if st.session_state.x_first_order_form != []:
                    for var in st.session_state.x_first_order_form: 
                        var_name = var.replace('{', '').replace('}', '')
                        st.session_state.prediction_var.append(var_name)
                        model_data_to_trans[var] = model_data[var_name]
                    x_firstOrder_data = model_data_to_trans[st.session_state.x_first_order_form]
                    final_data = pd.concat([y_data, x_firstOrder_data], axis=1)
                else:
                    final_data = y_data
                
                if st.session_state.x_second_order_form != []:
                    for var in st.session_state.x_second_order_form:
                        var_name = var.split("^")[0]
                        st.session_state.prediction_var.append(var_name)
                        model_data_to_trans[var] = model_data[var_name]**2
                    x_secondOrder_data = model_data_to_trans[st.session_state.x_second_order_form]
                    final_data = pd.concat([final_data, x_secondOrder_data], axis=1)

                if st.session_state.x_interaction_form != []:
                    for var in st.session_state.x_interaction_form:
                        var_name1 = var.split("*")[0].replace('{', '').replace('}', '')
                        var_name2 = var.split("*")[1].replace('{', '').replace('}', '')
                        st.session_state.prediction_var.extend([var_name1, var_name2])
                        model_data_to_trans[var] = model_data[var_name1].multiply(model_data[var_name2])
                    x_interaction_data = model_data_to_trans[st.session_state.x_interaction_form]
                    final_data = pd.concat([final_data, x_interaction_data], axis=1)
                
                if st.session_state.x_log_form != []:
                    for var in st.session_state.x_log_form:
                        var_name = var.split("(")[1].split(")")[0]
                        st.session_state.prediction_var.append(var_name)
                        model_data_to_trans[var] = np.log(model_data[var_name])
                    x_log_data = model_data_to_trans[st.session_state.x_log_form]
                    final_data = pd.concat([final_data, x_log_data], axis=1)
                
                if st.session_state.x_exp_form != []:
                    for var in st.session_state.x_exp_form:
                        var_name = var.split("(")[1].split(")")[0].replace('{', '').replace('}', '')
                        st.session_state.prediction_var.append(var_name)
                        model_data_to_trans[var] = np.exp(model_data[var_name])
                    x_exp_data = model_data_to_trans[st.session_state.x_exp_form]
                    final_data = pd.concat([final_data, x_exp_data], axis=1)

                if st.session_state.x_custom_form != []:
                    for var in st.session_state.x_custom_form:
                        var_name = var.split("^")[0]
                        st.session_state.prediction_var.append(var_name)
                        var_order = st.session_state.x_custom_order
                        model_data_to_trans[var] = model_data[var_name]**var_order
                    x_custom_data = model_data_to_trans[st.session_state.x_custom_form]
                    final_data = pd.concat([final_data, x_custom_data], axis=1)


                st.session_state.final_data = final_data
                
                
                

            # if st.session_state.final_data is not None:
                model_fitting = LinearRegression()
                X = st.session_state.final_data.iloc[:, 1:]
                Y = st.session_state.final_data.iloc[:, 0]
                model_fitting.fit(X, Y)
                beta_sklearn = np.insert(model_fitting.coef_, 0, model_fitting.intercept_)

                Y_varname = Y.name
                X_varname = X.columns

                # show estimated function and interpretation
                equation_est_mean = f"$E({Y_varname})$ = `{round(beta_sklearn[0], 2)}`"
                equation_est = f"${Y_varname}$ = `{round(beta_sklearn[0], 2)}`"
                func = ""
                interpretation = f"- This estimated regression function indicates that ：\n"
                for i, beta in enumerate(beta_sklearn[1:], start=1):
                    func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                    interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

                st.markdown(equation_est_mean+func)
                st.session_state.mean_est_function = equation_est_mean+func
                st.session_state.ols_function_interpre = interpretation
                with st.expander("See explanation"):
                    st.write(st.session_state.ols_function_interpre)

                #keep the function to next page
                func += " + $residuals$"
                st.session_state.est_function = equation_est+func

                y_hat = model_fitting.predict(X)
                fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
                PredictionErrorDisplay.from_predictions(
                    Y,
                    y_pred= y_hat,
                    kind="actual_vs_predicted",
                    ax=axs[0],
                    random_state=0,
                )
                axs[0].set_title("Actual vs. Predicted values")
                PredictionErrorDisplay.from_predictions(
                    Y,
                    y_pred=y_hat,
                    kind="residual_vs_predicted",
                    ax=axs[1],
                    random_state=0,
                )
                axs[1].set_title("Residuals vs. Predicted Values")
                fig.suptitle("Plotting cross-validated predictions")
                plt.tight_layout()
                st.pyplot(fig)

                if "independence_result" in st.session_state:
                    del st.session_state["independence_result"]
                if "variance_result" in st.session_state:
                    del st.session_state["variance_result"]
                if "normality_result" in st.session_state:
                    del st.session_state["normality_result"]
                if "linearity_result" in st.session_state:
                    del st.session_state["linearity_result"]
                if "dignostic_result" in st.session_state:
                    del st.session_state["dignostic_result"]
                if "wls_function" in st.session_state:
                    del st.session_state["wls_function"]
                if "wls_mean_function" in st.session_state:
                    del st.session_state["wls_mean_function"]
                if "wls_function_interpre" in st.session_state:
                    del st.session_state["wls_function_interpre"]
                if "bootstrap_results" in st.session_state:
                    del st.session_state["bootstrap_results"]
                if "wls_table1" in st.session_state:
                    del st.session_state["wls_table1"]
                if "wls_table2" in st.session_state:
                    del st.session_state["wls_table2"]
                if "wls_table_coefficients" in st.session_state:
                    del st.session_state["wls_table_coefficients"]
                if "weight" in st.session_state:
                    del st.session_state["weight"]
                if "sd_function_indep" in st.session_state:
                    del st.session_state["sd_function_indep"]
                if "residual_againstX" in st.session_state:
                    del st.session_state["residual_againstX"]
                if "beta" in st.session_state:
                    del st.session_state["beta"]
                if "beta_sd" in st.session_state:
                    del st.session_state["beta_sd"]
                

                


                




                
                
                


else:
    if 'error_text' in locals():
        st.error(error_text)
    else:
        st.error("Please back to 2_data_visualization page.")




pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/3_3️⃣Data_Filter.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/5__5️⃣Residual_Analysis.py")