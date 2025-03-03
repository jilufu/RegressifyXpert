import streamlit as st
from openai import OpenAI
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import math
import statsmodels.api as sm
from scipy.stats import f
import statsmodels.stats.api as sms
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# from collections import defaultdict
# import time


def chat_gpt():   
    st.sidebar.title("Regressitant")

    key = st.secrets["OPENAI_API_KEY"]

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        client = OpenAI(api_key=key)
        messages_container = st.container(height=200)
        with messages_container:
            for message in st.session_state.messages:
                st.chat_message(message["role"]).markdown(message["content"])
        
        prompt = st.chat_input("Any questions?")

        if prompt:
        # 顯示使用者訊息
            with messages_container:
                st.chat_message("user").markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            with messages_container:
                stream = client.chat.completions.create(
                                model=st.session_state["openai_model"],
                                messages=[
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state.messages
                                ],
                                stream=True,
                            )
                response = st.write_stream(stream)
                st.chat_message("assistant").markdown(response)
            
            # 保存 AI 助手回應
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()


def store_value(key):
    st.session_state[key] = st.session_state["_"+key]
def load_value(key):
    st.session_state["_"+key] = st.session_state[key]

def scatter_link_css():
    st.markdown(
        """
        <style>
        .st-emotion-cache-1ol4dec {
            padding: 0rem 0.75rem 1rem 0rem;
            background-color: unset;
            border: 0px;
        }
        .st-emotion-cache-1ol4dec:visited {
            color: rgb(46, 154, 255);
        }
        .st-emotion-cache-1ol4dec:active {
            color: rgb(46, 154, 255);
            background-color: unset;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def data_preprocessing_page_css():
    st.markdown(
        """
        <style>
        .st-emotion-cache-1ol4dec {
            padding: 0.9rem 0.75rem 0rem 0rem;
            background-color: unset;
            border: 0px;
        }
        .st-emotion-cache-1ol4dec:visited {
            color: rgb(46, 154, 255);
        }
        .st-emotion-cache-1ol4dec:active {
            color: rgb(46, 154, 255);
            background-color: unset;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def user_choose_model_vars(numerical_vars, categorical_vars):
    container_ModelFitting11 = st.container(border=True) 
    with container_ModelFitting11:
        st.write("<div style='padding-bottom: 0.5rem;'>Purpose of Data Analysis：</div>", unsafe_allow_html=True)
        container_ModelFitting1_2, container_ModelFitting1_3, \
        container_ModelFitting1_4, container_ModelFitting1_5 = st.columns([1.4,1,0.23,1.32])

        with container_ModelFitting1_2:
            st.write("Explore the Relationship between")

        with container_ModelFitting1_3:
            model_y = st.selectbox(options=numerical_vars, label="輸入y變數", index=None, placeholder="Y:Selecting a Dependent Variable",label_visibility="collapsed")
        with container_ModelFitting1_4:
            st.write("and")
        if model_y is not None:
            disable_x = False 
            varX = list(set(categorical_vars + numerical_vars) - {model_y})
        else:
            disable_x = True 
            varX = numerical_vars + categorical_vars  
        # 把model x存起來
        with container_ModelFitting1_5: 
            model_x = st.multiselect(options=varX, placeholder="X:Selecting Independent Variables",label="輸入x變數",label_visibility="collapsed",disabled=disable_x)

        return model_y, model_x

def scatter_explain(df):
    df_numeric = df[[st.session_state.user_choose_y]+ st.session_state.user_choose_x_num]
    if st.button("Scatterplot Matrix"):
        with st.spinner('Wait for it...'):
            scatter_matrix = sns.pairplot(df_numeric, diag_kind='kde')
            st.pyplot(scatter_matrix)
        # 解讀散步矩陣圖 描述是否有關係以集關係的強弱
        all_x = ", ".join(st.session_state.user_choose_x_num)
        text = ""
        for var in st.session_state.user_choose_x_num:
            text += f" Observe if there is a linear relationship between `{var}` and `{df_numeric.columns[0]}` . "

        st.markdown("Interpreting Scatterplot Matrix:")
        st.markdown(f"- Please note the linear relationship between the independent and dependent variables.{text} These relationships may suggest that `{all_x}` has a significant impact on predicting `{df_numeric.columns[0]}` ")
        if len(st.session_state.user_choose_x_num) > 1:
            st.markdown(f"- Please note the correlation between independent variables: observe if there is any relationship between the variables in `{all_x}` pair by pair.")
    
    return df_numeric

def handling_dummy(df):   
    need_to_dummy = []
    st.write("<div style='padding-bottom: 0.5rem;'>Dealing with Categorical Variables:</div>", unsafe_allow_html=True)
    for var in st.session_state.user_choose_x_cat:
        categories_level = df.loc[:, var].unique()
        categories_num = len(categories_level)
        if categories_num > 1:
            need_to_dummy.append(var)
            dummyvar = []
            for levels in categories_level[1:]:                
                new_var = f"{var}_{{{levels}}}"
                dummyvar.append(new_var)
                dummy_text = f" Converting them into dummy variables: ${'  '.join(dummyvar)}$."
        else:
            dummy_text = f"If there's only one category, the variable is not suitable for inclusion in the model."
                        
        st.write(f"- The number of categories for categorical variable `{var}` is `{categories_num}`, with values {categories_level}.{dummy_text}")
        if categories_num > 1:
            for dummy in dummyvar :
                parts = dummy.split("_")
                st.write(f"which ${dummy}$ is :")
                st.markdown(rf" $ {dummy} = \begin{{cases}} 1 & ,\;\; \text{{if }} {var} = {parts[1]} \\ 0 & ,\;\; \text{{otherwise}} \end{{cases}} $")

            for var in dummyvar:
                st.session_state.dummy_varName.append(var)
            

    if len(need_to_dummy) > 0:
        # st.session_state.dummy_varName = [item for sublist in st.session_state.dummy_varName for item in sublist]
        # create dummy variables in dataframe
        df_dummy = pd.get_dummies(df, columns=need_to_dummy, drop_first=False)
                    
        #只有dummy variables的dataframe
        only_dummy_df = df_dummy.loc[:, df_dummy.columns.difference(df.columns)]
    else:
        only_dummy_df = None
        need_to_dummy = None

    return only_dummy_df, need_to_dummy



def time_series_plot(e):
        fig, ax = plt.subplots()
        ax.plot(e)
        ax.set_xlabel("Observations order")
        ax.set_ylabel("Residuals")
        ax.set_title("Time Series Plot of Residuals")
        st.pyplot(fig)
def normal_probability_plot(e):
        fig, ax = plt.subplots()
        stats.probplot(e, dist="norm", plot=ax)
        ax.set_title("Normal Probability (Q-Q) Plot of Residuals")
        st.pyplot(fig)
def residuals_plot(Y_pred, e):
        fig, ax = plt.subplots()
        ax.scatter(Y_pred, e)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot")
        st.pyplot(fig)
def absolute_residuals_plot(Y_pred, e):
        fig, ax = plt.subplots()
        ax.scatter(Y_pred, np.abs(e))
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Absolute Residuals")
        ax.set_title("Absolute Residuals Plot")
        st.pyplot(fig)
def histogram_plot(e):
        mean = np.mean(e)
        std_dev = np.std(e)
        line_x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        normal_dist = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((line_x - mean) / std_dev) ** 2)
        fig, ax = plt.subplots()
        ax.hist(e, bins=20,density=True)
        ax.plot(line_x, normal_dist, color='red',label='Normal Distribution')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Histogram of Residuals")
        ax.legend()
        st.pyplot(fig)

def residual_against_x_plot(X,residual_againstX_plot,e):
    x = X[residual_againstX_plot]
    fig, ax = plt.subplots()
    ax.scatter(x, e)
    ax.axhline(y=0, color='red', linestyle='--')
    xlabel = residual_againstX_plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residuals")
    title = "Residuals Plot against " + xlabel
    ax.set_title(title)
    st.pyplot(fig)


def theLackOfFit(data,e):
    X_columns = data.columns[1:].to_list()
    Y_columns = data.columns[0]
    I = data.groupby(X_columns)[Y_columns].mean().reset_index().shape[0]
    k = len(X_columns)+1
    n = data.shape[0]
    mean_Y_by_X = data.groupby(X_columns)[Y_columns].transform('mean')
    data_lack = data.copy()
    data_lack['Y_mean'] = mean_Y_by_X
    data_lack['square_of_error'] = (data_lack[Y_columns] - data_lack['Y_mean']) ** 2
    ssef = data_lack['square_of_error'].sum()
    sser = np.sum(e ** 2)
    f_statistics = ((sser - ssef) / (I - k)) / (ssef / (n - I))
    lackOfFit_pValue = 1.0 - f.cdf(f_statistics, I-k, n-I)
    F_statistic = round(f_statistics,4)
    if lackOfFit_pValue < 0.0001:
        F_pvalue = "<0.0001"
    else:
        F_pvalue = math.floor(lackOfFit_pValue * 10**6) / 10**6
    st.write(f'F statistc: {F_statistic}')
    st.write(f'p-value: {F_pvalue}')
    # if lackOfFit_pValue < alpha:
    #     st.write('Reject the null hypothesis and the residuals are not linear tend.')
    # else:
    #     st.write('Fail to reject the null hypothesis and the residuals are linear tend.')
    return lackOfFit_pValue 



def Brown_Forsythe_Test(Y_pred,e, alpha):    
    # 計算 y_pred 的中位數
    median_y_pred = np.median(Y_pred)

    # 根據 y_pred 的中位數將 e 分成兩組
    #e_group1 = [e[i] for i in range(len(e)) if Y_pred[i] <= median_y_pred]
    e_group1 = [e[i] for i, pred in enumerate(Y_pred) if pred <= median_y_pred]
    e_group2 = [e[i] for i in range(len(e)) if Y_pred[i] > median_y_pred]

    BFresult = stats.levene(e_group1, e_group2, center='median')
    BFresult_statistic = round(BFresult.statistic,4)
    if BFresult.pvalue < 0.0001:
        BFresult_pvalue = "<0.0001"
    else:
        BFresult_pvalue = math.floor(BFresult.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {BFresult_statistic} ; p-value: {BFresult_pvalue}')
    #st.write(f'Brown-Forsythe Test p-value: {BFresult_pvalue}')
    if BFresult.pvalue < alpha:
        st.write(':red[Reject the null hypothesis and the variances are unequal]')
        check_box_heteroscedasticity = True
    else:
        st.write('Fail to reject the null hypothesis and the variances are equal')
        check_box_heteroscedasticity = False
    return check_box_heteroscedasticity

def Breusch_Pagan_test(X,e,alpha):
     x_copy = X.copy()   
     x_copy.insert(0, 'constant', 1.0)
     BPresult = sms.het_breuschpagan(e, x_copy)
     BPresult_statistic = round(BPresult[2],4)
     if BPresult[3] < 0.0001:
        BPresult_pvalue = "<0.0001"
     else:
        BPresult_pvalue = math.floor(BPresult[3] * 10**6) / 10**6
     st.write(f'F statistc: {BPresult_statistic} ; p-value: {BPresult_pvalue}')
     if BPresult[3] < alpha:
         st.write(':red[Reject the null hypothesis and the variances are unequal]')
         check_box_heteroscedasticity = True
     else:
         st.write('Fail to reject the null hypothesis and the variances are equal')
         check_box_heteroscedasticity = False
    
     return check_box_heteroscedasticity

def White_test(X, e,alpha):
    x_copy = X.copy()   
    x_copy.insert(0, 'constant', 1.0)
    White_result = sms.het_white(e, x_copy)
    White_result_statistic = round(White_result[2],4)
    if White_result[3] < 0.0001:
        White_result_pvalue = "<0.0001"
    else:
        White_result_pvalue = math.floor(White_result[3] * 10**6) / 10**6

    st.write(f'F statistc: {White_result_statistic} ; p-value: {White_result_pvalue}')
    if White_result[3] < alpha:
        st.write(':red[Reject the null hypothesis and the variances are unequal]')
        check_box_heteroscedasticity = True
    else:
        st.write('Fail to reject the null hypothesis and the variances are equal')
        check_box_heteroscedasticity = False
    return check_box_heteroscedasticity

def Shapiro_Wilk_Test(e,alpha_normal):
    shapiro_result = stats.shapiro(e)
    shapiro_result_statistic = round(shapiro_result.statistic,4)
    if shapiro_result.pvalue < 0.0001:
        shapiro_result_pvalue = "<0.0001"
    else:
        shapiro_result_pvalue = math.floor(shapiro_result.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {shapiro_result_statistic} ; p-value: {shapiro_result_pvalue}')
    if shapiro_result.pvalue < alpha_normal:
        st.write(':red[Reject the null hypothesis and the residuals are not normally distributed]')
        check_box_NonNormality = True
    else:
        st.write('Fail to reject the null hypothesis and the residuals are normally distributed')
        check_box_NonNormality = False
    return check_box_NonNormality

#perform Kolmogorov-Smirnov test for normality
def Kolmogorov_Smirnov_Test(e,alpha_normal):
    kstest_result = stats.kstest(e, 'norm')
    kstest_result_statistic = round(kstest_result.statistic,4)
    if kstest_result.pvalue < 0.0001:
        kstest_result_pvalue = "<0.0001"
    else:
        kstest_result_pvalue = math.floor(kstest_result.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {kstest_result_statistic} ; p-value: {kstest_result_pvalue}')
    if kstest_result.pvalue < alpha_normal:
        st.write(':red[Reject the null hypothesis and the residuals are not normally distributed]')
        check_box_NonNormality = True
    else:
        st.write('Fail to reject the null hypothesis and the residuals are normally distributed')
        check_box_NonNormality = False
    return check_box_NonNormality

def bootstrap_regression(data, n_bootstrap):
    coefficients = []
    # standard_errors = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot)
        model_boot_coef = np.insert(model_boot.coef_, 0, model_boot.intercept_)
        coefficients.append(model_boot_coef)
        #standard_errors.append(np.sqrt(np.mean((model_boot.predict(X_boot) - y_boot) ** 2)))
    return np.array(coefficients)

def wls_bootstrap_regression(data, n_bootstrap , sd_function_indep):
    coefficients = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model_origin = LinearRegression()
        model_origin.fit(X_boot, y_boot)
        Y_boot_pred = model_origin.predict(X_boot)
        e_boot = y_boot - Y_boot_pred
        e_boot = e_boot.rename('e')

        abs_e_boot = abs(e_boot).values.reshape(-1, 1)
        sd_boot_model = LinearRegression()
        if sd_function_indep == 'fitted value of Y':
            x_sdf = pd.Series(Y_boot_pred, name='Y_pred')
        else:
            x_sdf = X_boot[sd_function_indep]
        x_sdf = x_sdf.values.reshape(-1, 1)
        sd_boot_model.fit(x_sdf, abs_e_boot)
        s_i = sd_boot_model.predict(x_sdf)
        s_i_squared = s_i**2
        w_i = 1/s_i_squared
        W = np.diag(w_i.flatten())
                    
        # 計算 WLS 估計值
        wls_X = sm.add_constant(X_boot)
        wls_X_matrix = wls_X.values
        Y_matrix = y_boot.values
        X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
        model_boot_coef = X_transpose_X_inv @ X_transpose_Y
        coefficients.append(model_boot_coef)
    return np.array(coefficients)
    

def OLS_sd(X,Y,e):
    # 計算殘差的方差（均方誤差）
    n = len(Y)  # 樣本數量
    p = X.shape[1]  # 自變數數量
    residual_sum_of_squares = np.sum(e**2)
    residual_variance = residual_sum_of_squares / (n - p - 1)

    # 構建設計矩陣 X，並在最前面加上一列全為1的常數項，以考慮截距
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # 計算設計矩陣的偽逆（(X^T * X)^(-1)）
    X_with_intercept_T_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)

    # 計算係數的標準誤
    standard_errors = np.sqrt(np.diag(residual_variance * X_with_intercept_T_X_inv))

    return standard_errors

def extract_ols_regression_stats(model, data):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Dep_var = data.columns[0]
    ModelName = "OLS Regression"
    Method = "Least Squares"
    Sample_size = data.shape[0]
    Y_pred = model.predict(X)
    MSE = mean_squared_error(Y, Y_pred)
    MSE_adjusted = MSE*Sample_size/(Sample_size - (X.shape[1] + 1))
    RMSE_adjusted = np.sqrt(MSE_adjusted)
    R_squared = model.score(X, Y)
    Adj_R_squared = 1 - (1 - R_squared) * (Sample_size - 1) / (Sample_size - X.shape[1] - 1)
    F_statistic = (R_squared / (1 - R_squared)) * ((Sample_size - X.shape[1] - 1) / X.shape[1])
    Prob_F_statistic = 1 - stats.f.cdf(F_statistic, X.shape[1], Sample_size - X.shape[1] - 1)
    AIC = Sample_size * np.log(MSE) + 2 * (X.shape[1] + 1)
    BIC = Sample_size * np.log(MSE) + (X.shape[1] + 1) * np.log(Sample_size)
    regression_stats1 = {
        "Dependent Variable:": Dep_var,
        "Model:": ModelName,
        "Method:": Method,
        "Sample Size:": Sample_size,
        "MSE:" :  MSE_adjusted,
        "sqrt(MSE):": RMSE_adjusted
        }
    regression_stats2 = {
        "R-squared:": R_squared,
        "Adjusted R-squared:": Adj_R_squared,
        "F-statistic:": F_statistic,
        "Prob(F-statistic):": Prob_F_statistic,
        "AIC:": AIC,
        "BIC:": BIC   }
    return regression_stats1, regression_stats2

def extract_wls_regression_stats(residuals, data, MSEw):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Dep_var = data.columns[0]
    ModelName = "WLS Regression"
    Method = "Weighted Least Squares"
    Sample_size = data.shape[0]
    SSE = np.sum(residuals**2)
    MSE = SSE / (Sample_size - X.shape[1] - 1)
    RMSE_adjusted = np.sqrt(MSE)
    R_squared = 1 - SSE / np.var(Y)
    Adj_R_squared = 1 - (1 - R_squared) * (Sample_size - 1) / (Sample_size - X.shape[1] - 1)
    F_statistic = (R_squared / (1 - R_squared)) * ((Sample_size - X.shape[1] - 1) / X.shape[1])
    Prob_F_statistic = 1 - stats.f.cdf(F_statistic, X.shape[1], Sample_size - X.shape[1] - 1)
    AIC = Sample_size * np.log(SSE/Sample_size) + 2 * (X.shape[1] + 1)
    BIC = Sample_size * np.log(SSE/Sample_size) + (X.shape[1] + 1) * np.log(Sample_size)
    regression_stats1 = {
        "Dependent Variable:": Dep_var,
        "Model:": ModelName,
        "Method:": Method,
        "Sample Size:": Sample_size,
        "MSE:" :  MSE,
        "sqrt(MSE):": RMSE_adjusted,
        "MSEw: ":  MSEw }
    regression_stats2 = {
        "R-squared:": R_squared,
        "Adjusted R-squared:": Adj_R_squared,
        "F-statistic:": F_statistic,
        "Prob(F-statistic):": Prob_F_statistic,
        "AIC:": AIC,
        "BIC:": BIC   }
    return regression_stats1, regression_stats2

def coefficients_stats(beta, beta_sd, data ,coefficients_names):
    t_statistic = np.abs(beta / beta_sd)
    p_value = 2 * (1 - stats.t.cdf(t_statistic, data.shape[0] - data.shape[1] - 1))
    if data.shape[0] < 30 :
        degree_of_freedom = data.shape[0] - data.shape[1] - 1
        t_value = stats.t.ppf(1 - 0.05/2, degree_of_freedom)
    else:
        t_value = 1.96
    upper_bound = beta + t_value * beta_sd
    lower_bound = beta - t_value * beta_sd
    coefficients_stats = {
        "Variable": coefficients_names,
        "Coefficient": beta.tolist(),
        "Standard Error": beta_sd.tolist(),
        "|t|-statistic": t_statistic.tolist(),
        "Pr>|t|": p_value.tolist(),
        "95% Confidence Interval Lower Bound": lower_bound.tolist(),
        "95% Confidence Interval Upper Bound": upper_bound.tolist()}
    return coefficients_stats


def calculate_min_max_medium(df, var):
    """
    计算数据框特定列的最小值、最大值和中位数（实际存在的中间值）。
    
    参数:
    df (pd.DataFrame): 数据框
    var (str): 列名
    
    返回:
    tuple: 最小值、最大值和中位数
    """
    min_val = df[var].min()
    max_val = df[var].max()
    
    # 手动计算中位数（取排序后的中间值）
    sorted_vals = df[var].sort_values().values
    mid_index = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        # 取中间两个值中的较小一个
        median_val = sorted_vals[mid_index - 1]
    else:
        median_val = sorted_vals[mid_index]
    
    return min_val, max_val, median_val


def prediction_interval(data, x0 ):
  ''' Compute a prediction interval around the model's prediction of x0.

  INPUT
    model
      A predictive model with `fit` and `predict` methods
    X_train: numpy array of shape (n_samples, n_features)
      A numpy array containing the training input data
    y_train: numpy array of shape (n_samples,)
      A numpy array containing the training target data
    x0
      A new data point, of shape (n_features,)
    alpha: float = 0.05
      The prediction uncertainty

  OUTPUT
    A triple (`lower`, `pred`, `upper`) with `pred` being the prediction
    of the model and `lower` and `upper` constituting the lower- and upper
    bounds for the prediction interval around `pred`, respectively. '''
  

  alpha = 0.05
  # The authors choose the number of bootstrap samples as the square root
  # of the number of samples
  nbootstraps = 1000

  # Compute the m_i's and the validation residuals
  bootstrap_preds, val_residuals = np.empty(nbootstraps), []
  for b in range(nbootstraps):
    bootstrap_sample = resample(data, replace=True, n_samples=len(data))
    X_boot = bootstrap_sample.iloc[:, 1:]
    y_boot = bootstrap_sample.iloc[:, 0]
    model = LinearRegression()
    model.fit(X_boot, y_boot)
    preds = model.predict(X_boot)
    val_residuals.append(y_boot - preds)
    bootstrap_preds[b] = model.predict(x0)
  bootstrap_preds -= np.mean(bootstrap_preds)
  val_residuals = np.concatenate(val_residuals)

  # Compute the prediction and the training residuals
  X = data.iloc[:, 1:]
  y = data.iloc[:, 0]
  model_pred = LinearRegression()
  model_pred.fit(X, y)
  preds = model_pred.predict(X)
  train_residuals = y - preds

  # Take percentiles of the training- and validation residuals to enable
  # comparisons between them
  val_residuals = np.percentile(val_residuals, q = np.arange(100))
  train_residuals = np.percentile(train_residuals, q = np.arange(100))

  # Compute the .632+ bootstrap estimate for the sample noise and bias
  no_information_error = np.mean(np.abs(np.random.permutation(y) - \
    np.random.permutation(preds)))
  generalisation = np.abs(val_residuals.mean() - train_residuals.mean())
  no_information_val = np.abs(no_information_error - train_residuals)
  relative_overfitting_rate = np.mean(generalisation / no_information_val)
  weight = .632 / (1 - .368 * relative_overfitting_rate)
  residuals = (1 - weight) * train_residuals + weight * val_residuals

  # Construct the C set and get the percentiles
  C = np.array([m + o for m in bootstrap_preds for o in residuals])
  qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
  percentiles = np.percentile(C, q = qs)

  return percentiles[0], model_pred.predict(x0), percentiles[1]

def bootstrap_prediction(data, x0):
    n_iterations = 1000
    predictions = np.zeros((n_iterations, len(x0)))
    for i in range(n_iterations):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        predictions[i, :] = model.predict(x0)

    # X = data.iloc[:, 1:]
    # y = data.iloc[:, 0]
    # model_pred = LinearRegression()
    # model_pred.fit(X, y)
    # preds = model_pred.predict(x0)
    lower_percentile = 2.5
    upper_percentile = 97.5
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    center = np.mean(predictions, axis=0)
    # new_lower_bound = lower_bound - np.sqrt(MSE)
    # new_upper_bound = upper_bound + np.sqrt(MSE)

    return center, lower_bound, upper_bound


def wls_bootstrap_prediction(data, x0, sd_function_indep):
    n_iterations = 1000
    predictions = np.zeros((n_iterations, len(x0)))
    for i in range(n_iterations):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model_origin = LinearRegression()
        model_origin.fit(X_boot, y_boot)
        Y_boot_pred = model_origin.predict(X_boot)
        e_boot = y_boot - Y_boot_pred
        e_boot = e_boot.rename('e')

        abs_e_boot = abs(e_boot).values.reshape(-1, 1)
        sd_boot_model = LinearRegression()
        if sd_function_indep == 'fitted value of Y':
            x_sdf = pd.Series(Y_boot_pred, name='Y_pred')
        else:
            x_sdf = X_boot[sd_function_indep]
        x_sdf = x_sdf.values.reshape(-1, 1)
        sd_boot_model.fit(x_sdf, abs_e_boot)
        s_i = sd_boot_model.predict(x_sdf)
        s_i_squared = s_i**2
        w_i = 1/s_i_squared
        W = np.diag(w_i.flatten())
                    
        # 計算 WLS 估計值
        wls_X = sm.add_constant(X_boot)
        wls_X_matrix = wls_X.values
        Y_matrix = y_boot.values
        X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
        model_boot_coef = X_transpose_X_inv @ X_transpose_Y
        predictions[i, :] = x0 @ model_boot_coef[1:] + model_boot_coef[0]
        


    
    lower_percentile = 2.5
    upper_percentile = 97.5
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    center = np.mean(predictions, axis=0)
    

    return center, lower_bound, upper_bound