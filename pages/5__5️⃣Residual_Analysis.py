import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import boxcox
import statsmodels.api as sm
from util import chat_gpt, time_series_plot, residuals_plot, absolute_residuals_plot, normal_probability_plot, histogram_plot\
         , residual_against_x_plot, theLackOfFit, Brown_Forsythe_Test, Breusch_Pagan_test, White_test, Shapiro_Wilk_Test, Kolmogorov_Smirnov_Test\
         , load_value, store_value, extract_ols_regression_stats, coefficients_stats, wls_bootstrap_regression, extract_wls_regression_stats, bootstrap_regression, OLS_sd

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




if "beta" not in st.session_state:
    st.session_state.beta = None
if "beta_sd" not in st.session_state:
    st.session_state.beta_sd = None
if "ols_table1" not in st.session_state:
    st.session_state.ols_table1 = None
if "ols_table2" not in st.session_state:
    st.session_state.ols_table2 = None
if "ols_table_coefficients" not in st.session_state:
    st.session_state.ols_table_coefficients = None

if "dignostic_result" not in st.session_state:
    st.session_state.dignostic_result = False


st.header("Residual Analysis")

if st.session_state.final_data is not None:
    data = st.session_state.final_data

    st.subheader("The model is :")
    st.markdown(st.session_state.est_function)
    model = LinearRegression()
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    model.fit(X, Y)
    Y_pred = model.predict(X)
    e = Y - Y_pred
    e = e.rename('e')
    beta = np.insert(model.coef_, 0, model.intercept_)
    beta_sd = OLS_sd(X,Y,e)
    coefficients_names = ['Intercept'] + X.columns.tolist()

    st.session_state.ols_table1, st.session_state.ols_table2 = extract_ols_regression_stats(model, data)
    st.session_state.ols_table_coefficients = coefficients_stats(beta, beta_sd, data ,coefficients_names)
    
    
    st.subheader("Diagnostics and Formal Tests for the Assumptions")
    #st.write("1. Independence")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>1. &nbsp;&nbsp;Independence</div>", unsafe_allow_html=True)
    col1_1,col1_2 = st.columns(2)
    with col1_1:
        st.write("- Plot Diagnostic for Independence")
        time_series_plot(e)
    with col1_2:
        #st.write("To show the plot that do not violate the assumption of independence.")
        #st.link_button("Standard Time Series Plot", "https://streamlit.io/gallery")
        st.link_button("How to identify time series plot", "https://regressifyxpert.github.io/test/RESIDUAL.html#section-2")
        st.write("- Test for Independence")
        st.markdown("$H_0 : $ There is no correlation among the residuals.")
        st.markdown("$H_1 :$ The residuals are autocorrelated.")
        dw_statistics = round(durbin_watson(e),4)
        st.write("- Durbin-Watson Test")
        st.write(f"Durbin-Watson statistics: {dw_statistics}")
        if dw_statistics < 1.5:
            st.write(":red[It suggests that residuals may be positive serial correlation.]")
            check_box_Nonindependence = True
        elif dw_statistics > 2.5:
            st.write(":red[It suggests that residuals may be negative serial correlation.]")
            check_box_Nonindependence = True
        else:
            st.write("It suggests that residuals are independent.")
            check_box_Nonindependence = False

    #st.write("2. Linearity")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>2. &nbsp;&nbsp; Linearity</div>", unsafe_allow_html=True)
    col2_1,col2_2 = st.columns(2)
    with col2_1:
        st.write("- Plot Diagnostic for Linearity")
        residuals_plot(Y_pred, e)
    with col2_2:
        st.link_button("How to identify the residual plot", "https://regressifyxpert.github.io/test/RESIDUAL.html#section-3")
        st.write("- Test for Ascertaining the Linear Function")
        st.markdown("$H_0 : linear \;\;tendency$")
        st.markdown("$H_1 : non \;\; linear \;\; tendency$")
        # f test for lack of fit
        col2_2_1,col2_2_2 = st.columns([0.6,0.4])
        with col2_2_2 :
            alpha = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_linear")
        with col2_2_1 :
            st.write("- F Test for Lack of Fit")
            lackOfFit_pValue = theLackOfFit(data,e)
        if lackOfFit_pValue < alpha:
            st.write(':red[Reject the null hypothesis and the residuals are not linear tend.]')
            check_box_NonLinearity = True
        else:
            st.write('Fail to reject the null hypothesis and the residuals are linear tend.')
            check_box_NonLinearity = False




    #st.write("3. Equal Variance")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>3. &nbsp;&nbsp; Equal Variance</div>", unsafe_allow_html=True)
    col3_1,col3_2 = st.columns(2)
    with col3_1:
        st.write("- Plot Diagnostic for Constant Error Variance")
        absolute_residuals_plot(Y_pred, e)
    with col3_2:
        st.write("- Test for Constancy of Error Variance")
        st.markdown("$H_0 : Var(\epsilon_i)=\sigma^2$")
        st.markdown(r"$H_1 : Var(\epsilon_i)\neq\sigma^2$")
        col3_2_1,col3_2_2 = st.columns([0.6,0.4])
        col3_2_1.selectbox("Select the test ", ['Brown-Forsythe Test',"Breusch-Pagan test", "White test"], key="var_test")
        with col3_2_2 :
            alpha = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_variance")
        if st.session_state.var_test == "Brown-Forsythe Test":
            check_box_heteroscedasticity = Brown_Forsythe_Test(Y_pred,e, alpha)
        elif st.session_state.var_test == "Breusch-Pagan test":
            check_box_heteroscedasticity = Breusch_Pagan_test(X, e,alpha)   
        else:
            check_box_heteroscedasticity = White_test(X, e,alpha)     
    
    #st.write("4. Normality")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>4. &nbsp;&nbsp; Normality</div>", unsafe_allow_html=True)
    col4_1,col4_2 = st.columns(2)
    with col4_1:
        st.write("- Plot Diagnostic for Normality")
        tab1, tab2 = st.tabs(["Q-Q Plot", "Histogram"])
        with tab1:
            normal_probability_plot(e)
        with tab2:
            histogram_plot(e)
    with col4_2:
        st.link_button("How to identify the Q-Q plot", "https://regressifyxpert.github.io/test/RESIDUAL.html#section-5")
        st.write("- Test for Normality")
        st.markdown("$H_0 : \epsilon_i \sim Normal$")
        st.markdown("$H_1 : \epsilon_i \sim Non\;Normal$")
        col4_2_1,col4_2_2 = st.columns([0.6,0.4])
        col4_2_1.selectbox("Select the test ", ["Shapiro-Wilk Test","Kolmogorov-Smirnov Test"], key="noraml_test")
        with col4_2_2 :
            alpha_normal = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_normal")
        if st.session_state.noraml_test == "Shapiro-Wilk Test":
            check_box_NonNormality = Shapiro_Wilk_Test(e,alpha_normal)
        else:
            check_box_NonNormality = Kolmogorov_Smirnov_Test(e,alpha_normal)
    
    st.session_state.dignostic_result = True
    # show the residual diagnostic results
    if len(e)<30 :
        st.info("Please mainly refer to residual plots for the residual diagnosis")
    
    if "independence_result" not in st.session_state:
        st.session_state.independence_result = check_box_Nonindependence
   
    if "linearity_result" not in st.session_state:
        st.session_state.linearity_result = check_box_NonLinearity
    
    if "variance_result" not in st.session_state:
        st.session_state.variance_result = check_box_heteroscedasticity
    
    if "normality_result" not in st.session_state:
        st.session_state.normality_result = check_box_NonNormality
   

    container_residual_test = st.container(border=True) 
    with container_residual_test:
        st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'> Residual Diagnostic Results : </div>", unsafe_allow_html=True)
        rd_result1, rd_result2, rd_result3, rd_result4 = st.columns(4)
        load_value("independence_result")
        rd_result1.checkbox("NonIndependence of residuals",key="_independence_result", on_change=store_value,args=["independence_result"])
        load_value("linearity_result")
        rd_result2.checkbox("NonLinearity of residuals",key="_linearity_result", on_change=store_value,args=["linearity_result"])
        load_value("variance_result")
        rd_result3.checkbox("Heteroscedasticity of residuals",key="_variance_result", on_change=store_value,args=["variance_result"])
        load_value("normality_result")
        rd_result4.checkbox("NonNormality of residuals",key="_normality_result", on_change=store_value,args=["normality_result"])
      

    # Remedial Measures    

    if "wls_function" not in st.session_state:
        st.session_state.wls_function = None
    if "wls_mean_function" not in st.session_state:
        st.session_state.wls_mean_function = None
    if "wls_function_interpre" not in st.session_state:
        st.session_state.wls_function_interpre = None
    if "bootstrap_results" not in st.session_state:
        st.session_state.bootstrap_results = None
    if "wls_table1" not in st.session_state:
        st.session_state.wls_table1 = None
    if "wls_table2" not in st.session_state:
        st.session_state.wls_table2 = None
    if "wls_table_coefficients" not in st.session_state:
        st.session_state.wls_table_coefficients = None
    if "weight" not in st.session_state:
        st.session_state.weight = None
    
   
    sd_option = X.columns.tolist()
    sd_option.insert(0, 'fitted value of Y')
    if "sd_function_indep" not in st.session_state:
        st.session_state.sd_function_indep = 'fitted value of Y' 
    if "residual_againstX" not in st.session_state:
        st.session_state.residual_againstX = X.columns[0]
        
    
    if st.session_state.independence_result or st.session_state.linearity_result or st.session_state.variance_result or st.session_state.normality_result:
    
        st.subheader("Remedial Measures")
        #不獨立
        if st.session_state.independence_result:
            st.error("Beacuse the error terms are not independent, please go back to the last page to select all correlated variables. Otherwise, please use the Bootstrap method to estimate all parameter intervals.")
            #不線性
            if st.session_state.linearity_result:
                st.error("Because the residuals are not linear tend, please go back to the last page to reselect the independent form or add more correlated independent variables. Otherwise, the coefficients estimated by the model will be biased.")
            else :

                #異方差
                if st.session_state.variance_result:
                    # WLS
                    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Unequal Error Variances Remedial Measures - Weighted Least Squares</div>", unsafe_allow_html=True)
                    remedy_container2_1,remedy_container2_2 = st.columns([0.5,0.5])
                    remedy_container2_1.link_button("What is Weighted Least Squares ?", "https://regressifyxpert.github.io/test/problem.html#section-5")
                    remedy_container2_1.info("Use the residual plot against Xs or Y_hat to check the standard deviation function.")
                    remedy_container2_1.write(r"1. standard deviation function : regress $|e|$ against $X$ or $\hat{Y}$")
                    remedy_container2_1.write(r"2. the estimated weights : $w_i = \frac{1}{\hat{s_i}^2}$ where $\hat{s_i}$ is fitting value from  standard deviation function")
                    remedy_container2_1.latex(r'\hat{\beta} = (X^T W X)^{-1} X^T W y')
                    with remedy_container2_2 :
                        load_value("residual_againstX")
                        st.selectbox("Select X variable:", options= X.columns, key="_residual_againstX",label_visibility='collapsed',on_change=store_value,args=["residual_againstX"])
                        residual_against_x_plot(X,st.session_state.residual_againstX,e)
                        

                    remedy_container2_3,remedy_container2_4 = st.columns([0.5,0.5])
                    load_value("sd_function_indep")
                    with remedy_container2_3:   
                        st.selectbox("Select indep. variable of standard deviation function", options=sd_option, key="_sd_function_indep",on_change=store_value,args=["sd_function_indep"])
                        

                    if remedy_container2_4.button("Estimating model coefficients by WLS"):
                        abs_e = abs(e).values.reshape(-1, 1)
                        sd_model = LinearRegression()
                        if st.session_state.sd_function_indep == 'fitted value of Y':
                            x_sdf = pd.Series(Y_pred, name='Y_pred')
                        else:
                            x_sdf = X[st.session_state.sd_function_indep]
                        x_sdf = x_sdf.values.reshape(-1, 1)
                        sd_model.fit(x_sdf, abs_e)
                        s_i = sd_model.predict(x_sdf)
                        s_i_squared = s_i**2
                        w_i = 1/s_i_squared
                        W = np.diag(w_i.flatten())
                        st.session_state.weight = W

                        wls_X = sm.add_constant(X)
                        wls_X_matrix = wls_X.values
                        Y_matrix = Y.values
                        X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
                        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
                        X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
                        beta = X_transpose_X_inv @ X_transpose_Y
                        # 計算殘差
                        residuals = Y_matrix - wls_X_matrix @ beta

                        # 計算加權殘差的方差
                        n = len(Y)
                        p = wls_X.shape[1]
                        residual_sum_of_squares = np.sum(w_i.flatten() * residuals**2)
                        residual_variance = residual_sum_of_squares / (n - p)

                        # 計算係數的標準誤
                        beta_sd = np.sqrt(np.diag(residual_variance * X_transpose_X_inv))

                        
                        st.session_state.beta = beta
                        st.session_state.beta_sd = beta_sd

                        # write the estimated function
                        Y_varname = Y.name
                        X_varname = X.columns
                        # show estimated function and interpretation
                        equation_est_mean = f"$E({Y_varname})$ = `{round(beta[0], 2)}`"
                        equation_est = f"${Y_varname}$ = `{round(beta[0], 2)}`"
                        func = ""
                        interpretation = f"- This estimated regression function indicates that ：\n"
                        for i, beta in enumerate(beta[1:], start=1):
                            func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                            interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

                        st.session_state.wls_mean_function = equation_est_mean+func
                        
                        st.session_state.wls_function_interpre = interpretation

                        #keep the function to next page
                        func += " + $residuals$"
                        st.session_state.wls_function = equation_est+func

                        st.session_state.wls_table1, st.session_state.wls_table2 = extract_wls_regression_stats(residuals, data,residual_variance)
                        st.session_state.wls_table_coefficients = coefficients_stats(beta, beta_sd, data ,coefficients_names)

                    if st.session_state.wls_mean_function is not None:
                        st.write("The WLS estimated function of the mean response is as follows:")
                        st.markdown(st.session_state.wls_mean_function)

                # broostraping
                st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Boostraping for estimating parameter interval</div>", unsafe_allow_html=True)
                
                st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")

                
                if st.button("Show the Bootstraping Results"):
                    if st.session_state.wls_mean_function is not None :
                        bootstrap_coefficients = wls_bootstrap_regression(data, n_bootstrap=1000 , sd_function_indep=st.session_state.sd_function_indep)
                        Estimate_name = "WLS Estimate"
                        Standard_Error = "WLS Standard Error"
                    else:
                        bootstrap_coefficients = bootstrap_regression(data, n_bootstrap=1000)
                        Estimate_name = "OLS Estimate"
                        Standard_Error = "OLS Standard Error"
                        
                    if st.session_state.beta is None:
                        st.session_state.beta = beta
                    if st.session_state.beta_sd is None:
                        st.session_state.beta_sd = beta_sd

                    st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")
                    confidence_interval_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
                    confidence_interval_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)
                        
                    bootstrap_results = pd.DataFrame({
                        'Variable': coefficients_names,
                        Estimate_name : st.session_state.beta,
                        Standard_Error : st.session_state.beta_sd,
                        'Bootstrap Lower Bound': confidence_interval_lower,
                        'Bootstrap Upper Bound': confidence_interval_upper
                    })
                    st.session_state.bootstrap_results = bootstrap_results
                        
                
                if st.session_state.bootstrap_results is not None:
                    st.write(st.session_state.bootstrap_results)


        
        else:
            if st.session_state.linearity_result:
                st.error("Because the residuals are not linear tend, please go back to the last page to reselect the independent form or add more correlated independent variables. Otherwise, the coefficients estimated by the model will be biased.")
            else :
                if st.session_state.normality_result:
                    if len(e)>1000 :
                        st.info("When the sample size is large, instead of applying the relevant remedies, we can use the central limit theorem to approximate a normal distribution of the parameter estimators.")

                if st.session_state.variance_result or st.session_state.normality_result:
                    # boxcox transformation
                    remedy_container1_1,remedy_container1_2 = st.columns([0.45,0.55])
                    remedy_container1_1.write("<div style='font-size: 1.3rem; font-weight: 600; padding-bottom: 5px;'>&bull; &nbsp;&nbsp; Box-Cox transformation of &nbsp; Y</div>", unsafe_allow_html=True)
                    remedy_container1_2.link_button("What is Box-Cox method ?","https://regressifyxpert.github.io/test/problem.html#section-6")

                    remedy_container1_3,remedy_container1_4 = st.columns([0.35,0.65])  
                    if remedy_container1_3.button("Apply box-cox transformation"):
                        transformed_data, best_lambda = boxcox(Y)
                        st.session_state.boxcox_y = transformed_data
                        st.session_state.boxcox_lambda = best_lambda
                    if st.session_state.boxcox_lambda is not None :
                        remedy_container1_4.write(f"the best lambda : {st.session_state.boxcox_lambda}") 
                        st.error("Please go back to last page to reselect the dependent form by BoxCox(Y)")
                if st.session_state.variance_result:
                    # WLS
                    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Unequal Error Variances Remedial Measures - Weighted Least Squares</div>", unsafe_allow_html=True)
                    remedy_container2_1,remedy_container2_2 = st.columns([0.5,0.5])
                    remedy_container2_1.link_button("What is Weighted Least Squares ?", "https://regressifyxpert.github.io/test/problem.html#section-5")
                    remedy_container2_1.info("Use the residual plot against Xs or Y_hat to check the standard deviation function.")
                    remedy_container2_1.write(r"1. standard deviation function : regress $|e|$ against $X$ or $\hat{Y}$")
                    remedy_container2_1.write(r"2. the estimated weights : $w_i = \frac{1}{\hat{s_i}^2}$ where $\hat{s_i}$ is fitting value from  standard deviation function")
                    remedy_container2_1.latex(r'\hat{\beta} = (X^T W X)^{-1} X^T W y')

                    with remedy_container2_2 :
                        load_value("residual_againstX")
                        st.selectbox("Select X variable:", options= X.columns, key="_residual_againstX",label_visibility='collapsed',on_change=store_value,args=["residual_againstX"])
                        residual_against_x_plot(X,st.session_state.residual_againstX,e)

                    remedy_container2_3,remedy_container2_4 = st.columns([0.5,0.5])

                    load_value("sd_function_indep")
                    with remedy_container2_3:   
                        st.selectbox("Select indep. variable of standard deviation function", options=sd_option, key="_sd_function_indep",on_change=store_value,args=["sd_function_indep"])
                        
                    if remedy_container2_4.button("Estimating model coefficients by WLS"):
                        st.session_state.bootstrap_results = None
                        abs_e = abs(e).values.reshape(-1, 1)
                        sd_model = LinearRegression()
                        if st.session_state.sd_function_indep == 'fitted value of Y':
                            x_sdf = pd.Series(Y_pred, name='Y_pred')
                        else:
                            x_sdf = X[st.session_state.sd_function_indep]
                        x_sdf = x_sdf.values.reshape(-1, 1)
                        sd_model.fit(x_sdf, abs_e)
                        s_i = sd_model.predict(x_sdf)
                        s_i_squared = s_i**2
                        w_i = 1/s_i_squared
                        W = np.diag(w_i.flatten())
                        st.session_state.weight = W
                        
                        # 計算 WLS 估計值
                        wls_X = sm.add_constant(X)
                        wls_X_matrix = wls_X.values
                        Y_matrix = Y.values
                        X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
                        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
                        X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
                        beta = X_transpose_X_inv @ X_transpose_Y

                        # 計算殘差
                        residuals = Y_matrix - wls_X_matrix @ beta

                        # 計算加權殘差的方差
                        n = len(Y)
                        p = wls_X.shape[1]
                        residual_sum_of_squares = np.sum(w_i.flatten() * residuals**2)
                        residual_variance = residual_sum_of_squares / (n - p)

                        # 計算係數的標準誤
                        beta_sd = np.sqrt(np.diag(residual_variance * X_transpose_X_inv))

                        
                        st.session_state.beta = beta
                        st.session_state.beta_sd = beta_sd

                        # write the estimated function
                        Y_varname = Y.name
                        X_varname = X.columns
                        # show estimated function and interpretation
                        equation_est_mean = f"$E({Y_varname})$ = `{round(beta[0], 2)}`"
                        equation_est = f"${Y_varname}$ = `{round(beta[0], 2)}`"
                        func = ""
                        interpretation = f"- This estimated regression function indicates that ：\n"
                        for i, beta in enumerate(beta[1:], start=1):
                            func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                            interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

                        st.session_state.wls_mean_function = equation_est_mean+func
                        
                        st.session_state.wls_function_interpre = interpretation

                        #keep the function to next page
                        func += " + $residuals$"
                        st.session_state.wls_function = equation_est+func

                        st.session_state.wls_table1, st.session_state.wls_table2 = extract_wls_regression_stats(residuals, data,residual_variance)
                        st.session_state.wls_table_coefficients = coefficients_stats(beta, beta_sd, data ,coefficients_names)


                    if st.session_state.wls_mean_function is not None:
                        st.write("The WLS estimated function of the mean response is as follows:")
                        st.markdown(st.session_state.wls_mean_function)



                if st.session_state.normality_result :
                    # broostraping
                    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Bootstraping for estimating parameter interval</div>", unsafe_allow_html=True)
                    
                    st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")

                    
                    if st.button("Show the Bootstraping Results"):
                        
                        if st.session_state.wls_mean_function is not None :
                            bootstrap_coefficients = wls_bootstrap_regression(data, n_bootstrap=1000, sd_function_indep=st.session_state.sd_function_indep)
                            Estimate_name = "WLS Estimate"
                            Standard_Error = "WLS Standard Error"
                        else:
                            bootstrap_coefficients = bootstrap_regression(data, n_bootstrap=1000)
                            Estimate_name = "OLS Estimate"
                            Standard_Error = "OLS Standard Error"


                        if st.session_state.beta is None:
                            st.session_state.beta = beta
                        if st.session_state.beta_sd is None:
                            st.session_state.beta_sd = beta_sd

                            
                        confidence_interval_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
                        confidence_interval_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)
                            
                        bootstrap_results = pd.DataFrame({
                            'Variable': coefficients_names,
                            Estimate_name : st.session_state.beta,
                            Standard_Error : st.session_state.beta_sd,
                            'Bootstrap Lower Bound': confidence_interval_lower,
                            'Bootstrap Upper Bound': confidence_interval_upper
                        })

                        st.session_state.bootstrap_results = bootstrap_results

                    if st.session_state.bootstrap_results is not None:   
                        st.write(st.session_state.bootstrap_results)
    else:
        st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&nbsp;&nbsp; Satisfying the four modelling assumptions, please go to the next page.</div>", unsafe_allow_html=True)           
            

else:
    st.error("Please back to model fitting page and select a model.")

# analysis of appropriateness of the model

# show residual plot against fitted values, x variables,  x interaction terms 
# A systematic pattern in this plot would suggest that an interaction effect may be present


#plot of the absolute residuals against the fitted values
# use WLS to fit the model
# conclusion - There is no indication of nonconstancy of the error varianCe


#normal probability plot of the residuals. 
# broostraping 
# inference and conclusion 
#The pattern is moderately linear. 
#The coefficient of correlation between the ordered residuals and their expected values 
#under normality is .980. This high value (the interpolated critical value in Table B.6 
#for n = 21 and ex = .05 is .9525) helps to confirm the reasonableness of the conclusion 
#that the error terms are fairly normally distributed.


pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/4_4️⃣Model_Fitting.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/6_6️⃣Model_Result.py")

