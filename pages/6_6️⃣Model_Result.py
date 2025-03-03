import streamlit as st
import pandas as pd
from util import chat_gpt, calculate_min_max_medium, prediction_interval ,bootstrap_prediction,wls_bootstrap_prediction
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
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

if "final_data" not in st.session_state:
    st.session_state.final_data = None
if "dignostic_result" not in st.session_state:
    st.session_state.dignostic_result = False

if "bootstrap_results" not in st.session_state:
    st.session_state.bootstrap_results = None
if "wls_mean_function" not in st.session_state:
    st.session_state.wls_mean_function = None

st.header("Regression Model Result")

if st.session_state.final_data is not None:
    if st.session_state.dignostic_result:
        if st.session_state.linearity_result:
            st.error("The linearity assumption is not satisfied.")
        else:
            st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; The estimated regression function :</div>", unsafe_allow_html=True)
            if st.session_state.wls_mean_function is None:
                Model_Informations_table = pd.DataFrame.from_dict(st.session_state.ols_table1, orient="index", columns=["Value"]).T
                Model_selection_table = pd.DataFrame.from_dict(st.session_state.ols_table2, orient="index", columns=["Value"]).T
                st.markdown(st.session_state.est_function)
                st.markdown(st.session_state.mean_est_function) 
                with st.expander("See explanation"):
                    st.write(st.session_state.ols_function_interpre)

                if st.session_state.bootstrap_results is  None :
                    coeff_table = pd.DataFrame(st.session_state.ols_table_coefficients).set_index("Variable")
                                        
                else:
                    coeff_table = st.session_state.bootstrap_results
                    

            else:
                Model_Informations_table = pd.DataFrame.from_dict(st.session_state.wls_table1, orient="index", columns=["Value"]).T
                Model_selection_table = pd.DataFrame.from_dict(st.session_state.wls_table2, orient="index", columns=["Value"]).T
                st.markdown(st.session_state.wls_function)
                st.markdown(st.session_state.wls_mean_function)
                with st.expander("See explanation"):
                    st.write(st.session_state.wls_function_interpre)
                if st.session_state.bootstrap_results is  None :       
                    coeff_table = pd.DataFrame(st.session_state.wls_table_coefficients).set_index("Variable")
                else:
                    coeff_table = st.session_state.bootstrap_results
                    


            # show all result tables
            Model_Informations1, Model_Informations2 = st.columns([0.33, 0.67])
            Model_Informations1.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Model Informations :</div>", unsafe_allow_html=True)
            Model_Informations2.link_button("More Detail....", "https://regressifyxpert.github.io/test/%E7%B5%B1%E8%A8%88%E6%8E%A8%E8%AB%96.html")
            st.dataframe(Model_Informations_table)

            Model_selection1, Model_selection2 = st.columns([0.42, 0.58])
            Model_selection1.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Criteria for Model Selection :</div>", unsafe_allow_html=True)
            Model_selection2.link_button("More Detail....", "https://regressifyxpert.github.io/test/%E7%B5%B1%E8%A8%88%E6%8E%A8%E8%AB%96.html#section-2")
            st.dataframe(Model_selection_table)
            
            coeff1, coeff2 = st.columns([0.22, 0.78])
            coeff1.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Coefficients :</div>", unsafe_allow_html=True)
            coeff2.link_button("More Detail....", "https://regressifyxpert.github.io/test/%E7%B5%B1%E8%A8%88%E6%8E%A8%E8%AB%96.html#section-4")
            
            st.dataframe(coeff_table)
            

            # prediction
            st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Prediction :</div>", unsafe_allow_html=True)
            
            if st.session_state.prediction_var != []:
                catvar_predict = []
                numvar_predict = []
                prediction_var = list(set(st.session_state.prediction_var))
                for var in prediction_var:
                    if "_" in var:
                        catvar_predict.append(var.split("_")[0])
                    else:
                        numvar_predict.append(var)

                if st.session_state.df_filter is None:
                    df = st.session_state.data_convert
                else:
                    df = st.session_state.df_filter

                user_dummy_vars = pd.DataFrame()
                user_numeric_vars = pd.DataFrame()

                prediction_container = st.container(border=True)
                with prediction_container:
                    st.write("Please enter the value of the following variables to predict the response variable.")
                    
                    #categorical input
                    if catvar_predict != []:
                        catvar_predict = list(set(catvar_predict))
                        st.write("Categorical variables:")
                        num_cols = 4
                        if len(catvar_predict) % num_cols == 0:
                            num_rows = len(catvar_predict) // num_cols
                        else:
                            num_rows = len(catvar_predict) // num_cols + 1

                        user_selections_cat = {}

                        for row in range(num_rows):
                            cols = st.columns(num_cols)
                            for col in range(num_cols):
                                var_idx = row * num_cols + col
                                if var_idx >= len(catvar_predict):
                                    break
                                var = catvar_predict[var_idx]
                                user_selections_cat[var] = cols[col].selectbox(var, df[var].unique(), key=var, index=0)
                        
                        # st.write(user_selections_cat)
                        dummy_vars = pd.get_dummies(df[catvar_predict])
                        # st.write(dummy_vars)
                        user_dummy_vars = pd.DataFrame(0, index=[0], columns=dummy_vars.columns)
                        for var, value in user_selections_cat.items():
                            user_dummy_vars[f"{var}_{value}"] = 1

                        new_columns = [f"{col.split('_')[0]}_{{{col.split('_')[1]}}}" for col in user_dummy_vars.columns]
                        user_dummy_vars.columns = new_columns
                        # st.write(user_dummy_vars)
                    
                    #number input
                    if numvar_predict != []:
                        numvar_predict = list(set(numvar_predict))
                        st.write("Numerical variables:")
                        num_cols = 4
                        if len(numvar_predict) % num_cols == 0:
                            num_rows = len(numvar_predict) // num_cols
                        else:
                            num_rows = len(numvar_predict) // num_cols + 1

                        user_selections_num = {}

                        for row in range(num_rows):
                            cols = st.columns(num_cols)
                            for col in range(num_cols):
                                var_idx = row * num_cols + col
                                if var_idx >= len(numvar_predict):
                                    break
                                var = numvar_predict[var_idx]
                                min_val, max_val, median_val = calculate_min_max_medium(df, var)
                                
                                cols[col].text_input(f'Input a number of {var}', value=median_val, key=var)
                                
                                
                                try:
                                    input_num = float(st.session_state[var])
                                except:
                                    st.error(f"Please enter a number of {var}.")
                                    input_num = None
                                    user_selections_num[var] = np.nan
                                    

                                if input_num is not None:                                    
                                    if input_num < min_val or input_num > max_val:
                                        st.error(f"Please enter {var} value between {min_val} and {max_val}.")
                                        user_selections_num[var] = np.nan
                                    else:
                                        user_selections_num[var] = input_num
                                        

                        
                        user_numeric_vars = pd.DataFrame(0, index=[0], columns=numvar_predict)
                        for var, value in user_selections_num.items():
                            user_numeric_vars[var] = value 


                    if not user_dummy_vars.empty and not user_numeric_vars.empty:
                        combined_vars = pd.concat([user_dummy_vars, user_numeric_vars], axis=1)
                        
                    elif not user_dummy_vars.empty:
                        combined_vars = user_dummy_vars
                        
                    elif not user_numeric_vars.empty:
                        combined_vars = user_numeric_vars
                        
                    else:
                        combined_vars = pd.DataFrame()

            

                if combined_vars.isnull().sum().sum() == 0:
                    pred_1,pred_2,pred_3 = st.columns([0.23,0.6,0.17])        
                    if pred_2.button("Show the interval of predicted response variable"):
                        combined_vars_to_trans = combined_vars.copy()

                       
                        if st.session_state.x_first_order_form != []:
                            xpred_firstOrder_data = combined_vars[st.session_state.x_first_order_form]
                            x_h = xpred_firstOrder_data
                            
                        if st.session_state.x_second_order_form != []:
                            for var in st.session_state.x_second_order_form:
                                var_name = var.split("^")[0]
                                combined_vars_to_trans[var] = combined_vars[var_name]**2
                            xpred_secondOrder_data = combined_vars_to_trans[st.session_state.x_second_order_form]
                            x_h = pd.concat([x_h, xpred_secondOrder_data], axis=1)

                        if st.session_state.x_interaction_form != []:
                            for var in st.session_state.x_interaction_form:
                                var1, var2 = var.split("*")
                                combined_vars_to_trans[var] = combined_vars[var1].multiply(combined_vars[var2])
                            xpred_interaction_data = combined_vars_to_trans[st.session_state.x_interaction_form]
                            x_h = pd.concat([x_h, xpred_interaction_data], axis=1)

                        if st.session_state.x_log_form != []:
                            for var in st.session_state.x_log_form:
                                var_name = var.split("log(")[1].split(")")[0]
                                combined_vars_to_trans[var] = np.log(combined_vars[var_name])
                            xpred_log_data = combined_vars_to_trans[st.session_state.x_log_form]
                            x_h = pd.concat([x_h, xpred_log_data], axis=1)

                        if st.session_state.x_exp_form != []:
                            for var in st.session_state.x_exp_form:
                                var_name = var.split("exp(")[1].split(")")[0]
                                combined_vars_to_trans[var] = np.exp(combined_vars[var_name])
                            xpred_exp_data =  combined_vars_to_trans[st.session_state.x_exp_form]
                            x_h = pd.concat([x_h, xpred_exp_data], axis=1)

                        if st.session_state.x_custom_form != []:
                            for var in st.session_state.x_custom_form:
                                var_name = var.split("^")[0]
                                var_order = st.session_state.x_custom_order
                                combined_vars_to_trans[var] = combined_vars[var_name]**var_order
                            xpred_custom_data =  combined_vars_to_trans[st.session_state.x_custom_form]
                            x_h = pd.concat([x_h, xpred_custom_data], axis=1)

                        x0 = x_h.copy()
                        x_h.insert(0, 'const', 1)
                        
                       
                        x_h_matrix = x_h.values
                     
                        final_data = st.session_state.final_data
                        X = final_data.iloc[:, 1:]
                        X = sm.add_constant(X)
                        X_matrix = X.values
                        # st.write(X_matrix)
                        MSE = Model_Informations_table['MSE:'][0]
                        
                                             
                        

                        if st.session_state.wls_mean_function is None :
                            if st.session_state.bootstrap_results is  None :
                                y_h = x_h.values @ coeff_table["Coefficient"].values
                                se_mean_y_h = np.sqrt(MSE * (x_h_matrix @ np.linalg.inv(X_matrix.T @ X_matrix) @ x_h_matrix.T))        
                        
                                if  Model_Informations_table['Sample Size:'][0] < 30:
                                    degree_of_freedom = Model_Informations_table['Sample Size:'][0] - coeff_table.shape[0]
                                    t_value = stats.t.ppf(1 - 0.05/2, degree_of_freedom)
                                else:
                                    t_value = 1.96

                                lower = y_h[0] - t_value * se_mean_y_h[0, 0]
                                upper = y_h[0] + t_value * se_mean_y_h[0, 0]

                                ols_mean_y_h_table = pd.DataFrame({
                                    "Mean Response": [y_h[0]],
                                    "Standard Error": [se_mean_y_h[0, 0]],
                                    "95% Confidence Interval Lower Bound": [lower],
                                    "95% Confidence Interval Upper Bound": [upper]
                                })
                                st.write(r"1. Interval Estimation of Mean Response $E{Y_h}$")
                                # st.write(r"$$ \hat{Y}_h \pm t_{\alpha/2, n-p} \times \sqrt{MSE} \times \sqrt{X_h^T(X^TX)^{-1}X_h} $$") 
                                st.write(r"The $95\%$ confidence interval for the mean response at the specified values of the independent variables is given by:")
                    
                                st.write(ols_mean_y_h_table)

                                se_new_y_h = np.sqrt(MSE + se_mean_y_h[0, 0]**2)
                                lower = y_h[0] - t_value * se_new_y_h
                                upper = y_h[0] + t_value * se_new_y_h

                                ols_new_y_h_table = pd.DataFrame({
                                    "New Observation": [y_h[0]],
                                    "Standard Error": [se_new_y_h],
                                    "95% Prediction Interval Lower Bound": [lower],
                                    "95% Prediction Interval Upper Bound": [upper]
                                })
                                
                                st.write(r"2. Prediction of New Observation $Y_{h(new)}$")
                                st.write(r"The $95\%$ prediction interval for a new observation $Y_{h(new)}$ at the specified values of the independent variables is given by:")

                                st.write(ols_new_y_h_table)

                            else:
                                st.write(r"The $95\%$ prediction interval of the mean response variable at the specified values of the independent variables is given by:")
                                mean,lowerb, upperb =bootstrap_prediction(final_data, x0)
                                ols_bootstrap_pred_table = pd.DataFrame({
                                    "Prediction": mean,
                                    "95% Prediction Interval Lower Bound": lowerb,
                                    "95% Prediction Interval Upper Bound": upperb
                                })
                                st.write(ols_bootstrap_pred_table)
                        else:
                            if st.session_state.bootstrap_results is  None :
                                
                                MSEw = Model_Informations_table['MSEw: '][0]
                                W = st.session_state.weight
                                se_mean_y_h = np.sqrt(MSEw * (x_h_matrix @ np.linalg.inv(X_matrix.T @ W @ X_matrix) @ x_h_matrix.T))
                                y_h = x_h.values @ coeff_table["Coefficient"].values

                                if  Model_Informations_table['Sample Size:'][0] < 30:
                                    degree_of_freedom = Model_Informations_table['Sample Size:'][0] - coeff_table.shape[0]
                                    t_value = stats.t.ppf(1 - 0.05/2, degree_of_freedom)
                                else:
                                    t_value = 1.96

                                lower = y_h[0] - t_value * se_mean_y_h[0, 0]
                                upper = y_h[0] + t_value * se_mean_y_h[0, 0]

                                wls_mean_y_h_table = pd.DataFrame({
                                    "Mean Response": [y_h[0]],
                                    "Standard Error": [se_mean_y_h[0, 0]],
                                    "95% Confidence Interval Lower Bound": [lower],
                                    "95% Confidence Interval Upper Bound": [upper]
                                })
                                st.write(r"- Interval Estimation of Mean Response $E{Y_h}$")
                                st.write(r"The $95\%$ confidence interval for the mean response at the specified values of the independent variables is given by:")
                    
                                st.write(wls_mean_y_h_table)
                            else:
                                st.write(r"The $95\%$ prediction interval of the mean response variable at the specified values of the independent variables is given by:")
                                mean,lowerb, upperb = wls_bootstrap_prediction(final_data, x0, st.session_state.sd_function_indep)
                                wls_bootstrap_pred_table = pd.DataFrame({
                                    "Prediction": mean,
                                    "95% Prediction Interval Lower Bound": lowerb,
                                    "95% Prediction Interval Upper Bound": upperb
                                })
                                st.write(wls_bootstrap_pred_table)

                        
            
            
            

            

            

                



    else:
        st.error("Please back to Residual Analysis page and Make a residual diagnosis.")

else:
    st.error("Please back to model fitting page and select a model.")



# 報表分析
    # analysis of variance
        # test of regression relation p.244
        # coefficients of multiple determination
        # 檢定個別變數

    # estimation and inference of regression parameters
    # 根據常態檢定選擇使用方法 中央極限定理或broostraping

# estimation of mean response

# prediction for new observation

pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/5__5️⃣Residual_Analysis.py")
