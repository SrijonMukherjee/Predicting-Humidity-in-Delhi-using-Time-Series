import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

#ADF test
def adf_test(x):
    adf_list = adfuller(x)
    print('\n ADF Statistic: %f' % adf_list[0])
    print('\n p-value: %f' % adf_list[1])
    print('\n Critical Values:')
    for key, value in adf_list[4].items():
        print('\t%s: %.3f' % (key, value))




#Correlation Coefficient
def correlation_coefficient_cal(x, y):
    num = 0
    denom_x_sq = 0
    denom_y_sq = 0

    for i in range(0, len(x)):
        a =  (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        num += a
    
    for i in range(len(x)):

            b = np.square(x[i] - np.mean(x))
            denom_x_sq += b
            c = np.square(y[i] - np.mean(y))
            denom_y_sq += c
    denom = np.sqrt(denom_x_sq) * np.sqrt(denom_y_sq)
    r = num / denom
    return r

#Partial Correlation
def partial_correlation(x, y, confound):
    num = correlation_coefficient_cal(x, y)  - (correlation_coefficient_cal(x, confound) * correlation_coefficient_cal(y, confound))
    den = np.sqrt(1 - np.square(correlation_coefficient_cal(x, confound))) * np.sqrt(1 - np.square(  correlation_coefficient_cal (y, confound))  )
    return num/den

#Partial Correlation Hypothesis Test
def hypothesis_test(number_samples, number_confounding, partial_correlation_value):
    alpha = 0.05
    degrees_freedom = number_samples - 2 - number_confounding
    critical_t_value = t.ppf(1-alpha, df = degrees_freedom)
    t_correlation = np.abs(partial_correlation_value * np.sqrt( degrees_freedom / (1 - np.square(  partial_correlation_value)  )))
    
    if t_correlation > critical_t_value:
        print("The t value for the hypothesis test is: {}".format(t_correlation))
        print("The t vale at the critical threshold of 0.05 is: {}".format(critical_t_value))
        print("Thus the correlation is statistically significant")
    else:
        print("The t value for the hypothesis test is: {}".format(t_correlation))
        print("The t vale at the critical threshold of 0.05 is: {}".format(critical_t_value))
        print("Thus the correlation is not significant")
    
    return t_correlation , critical_t_value

#Auto Correlation
def auto_correlation(x, k):
    num = 0
    denom = 0
    if k > 0:
        for i in range(k, len(x)): 
            a = (x[i] - np.mean(x)) * (x[i-k] - np.mean(x))
            num += a
        for i in range(0, len(x)):
            a = np.square(x[i] - np.mean(x))
            denom += a
    if k < 0:
        k = k * -1
        for i in range(k, len(x)): 
            a = (x[i] - np.mean(x)) * (x[i-k] - np.mean(x))
            num += a
        for i in range(0, len(x)):
            a = np.square(x[i] - np.mean(x))
            denom += a
    elif k == 0:
        return 1  
    acr = num/denom
    return acr

#Auto Correlation Plot

def acf_plot(number_lags, array_to_plot):
    lags = np.arange(-1 * (number_lags - 1), number_lags, 1)
    acf_list = []
    for i in lags:
        acf_list.append(auto_correlation(array_to_plot, i))
    plt.stem(lags, acf_list)
    plt.title("ACF Plot for y(t)")
    plt.xlabel("Lags")
    plt.show()

#Plot for acf
def theoretical_acf(lags, x):
    acf_list = []
    for i in range(lags):
        acf_list.append(auto_correlation(x,i))
    return acf_list
#GPac table
def gpac(theoretical_acf, j_value, k_value):
    gpac = np.zeros((j_value,k_value-1))
    for k in range(1,k_value):
        for j in range(0,j_value):
            den = np.zeros((k,k))
            for row in range(k):
                for col in range(k):
                    den[row][col] = theoretical_acf[abs(j+row-col)]
            num = den.copy()
            for row in range(k):
                num[row][k-1] = theoretical_acf[j+row+1]

            det_num = np.linalg.det(num)
            det_den = np.linalg.det(den)
            gpac[j][k-1] = det_num/det_den
    return gpac

#Stats Model ARMA
def arma_model_parameter_estimation(y,na,nb):
    model = sm.tsa.ARMA(y,(na,nb)).fit(trend = 'nc', disp = 0)
    for i in range(na):
         print("The AR coefficient a{}".format(i), "is", model.params[i])
         for i in range(nb):
             print("The MA coefficient b{}".format(i), "is", model.params[i+na])
    print(model.summary())  
    return model 

def auto_correlation_list(array_plot):
   acr_list = []
   for i in range(0, len(array_plot)):
       acr_list.append(auto_correlation(array_plot, i))
   return acr_list

#Qvalue
def find_Q_value(x, x_pred, lags):
   list_errors = x - x_pred
   acr_error_list = theoretical_acf(lags, list_errors)
   sum_square_ac = 0
   for i in range(1, len(acr_error_list)):
      sum_square_ac = sum_square_ac +  np.square(acr_error_list[i])
   Q = len(list_errors) * sum_square_ac
   return Q 


def average_method(x, k):
    x_pred = []
    for i in range(0, len(x)):
        if i == k:
            x_pred.append(np.mean(x[:k]))
            break
        else:
            x_pred.append(x[i])
    return x_pred

#Plot for average method: predicted and true values
def average_method_plot(x, x_pred):
    plt.plot(x_pred)
    plt.legend(["Original Values"])
    plt.plot(x[:len(x_pred)])
    plt.legend(["Predicted Model"])
    plt.title("Plot of Predicted Model and Actual Values")
    plt.show()

#Implementing the drift Method
def drift_method(x, k):
   x_pred = []
   for i in range(0, len(x)):
       if i == k:
           x_pred.append(x[i-1]+((x[i-1]-x[0])/(i-1)))
           break
       else:
           x_pred.append(x[i])
   return x_pred


#Implementing the Naive Method
def naive_method(x, k):
    x_pred = []
    for i in range(0, len(x)):
        if i == k:
            x_pred.append(x[i-1])
            break
        else:
            x_pred.append(x[i])
    return x_pred

#Plot for naive method: predicted and true values
def naive_method_plot(x, x_pred):
    plt.plot(x_pred)
    plt.legend(["Original Values"])
    plt.plot(x[:len(x_pred)])
    plt.legend(["Predicted Model"])
    plt.title("Plot of Predicted Model and Actual Values")
    plt.show()

def simple_exponential_method(x,  alpha, initial_condition):
    x_pred = []
    for i in range(1,len(x)):
      if i==1:
        x_pred.append((alpha*x[i-1])+initial_condition * (1-alpha))
      else:    
        x_pred.append((alpha*x[i-1]) + ((1-alpha)*x_pred[i-2]))
    x_pred.insert(0, x[0])
    return x_pred

#Plot for simple_exponential method: predicted and true values
def simple_exponential_method_plot(x, x_pred):
  plt.plot(x_pred)
  plt.legend(["Original Values"])
  plt.plot(x[:len(x_pred)])
  plt.legend(["Predicted Model"])
  plt.title("Plot of Predicted Model and Actual Values")
  plt.show()




#Writing function for forecast error for average and drift:
def forecast_error(x, x_pred):
   error = x[len(x_pred)-1] - x_pred[-1]
   return error

#Writing function for variance of errors:
def forecast_error_variance(x, *args):
    var_list = []
    for elem in args:
        var_list.append(forecast_error(x, elem))
    return np.var(var_list)


#Writing function for sum square errors:
def sum_square_error(x, *args):
    error_square = 0
    sum_error_square = 0
    for elem in args:
        for i in range(0, len(x)):
            if x[i] != elem[i]:
                sum_error_square  = sum_error_square + np.square(x[i] - elem[i])
                break
    return sum_error_square 


#Writing a function for MSE calculation:
def mean_squared_error(x, *args):
    error_square = 0
    sum_error_square = 0
    for elem in args:
        for i in range(0, len(x)):
            if x[i] != elem[i]:
                sum_error_square  = sum_error_square + np.square(x[i] - elem[i])
                break
    return (sum_error_square / len(args))

def find_Q_value(x, x_pred, lags):
   list_errors = list(x - x_pred)
   acr_error_list = theoretical_acf(lags, list_errors)
   sum_square_ac = 0
   for i in range(1, len(acr_error_list)):
      sum_square_ac = sum_square_ac +  np.square(acr_error_list[i])
   Q = len(list_errors) * sum_square_ac
   return Q 

def acf_plot_residuals(number_lags, array_to_plot):
    array_to_plot = list(array_to_plot)
    lags = np.arange(-1 * (number_lags - 1), number_lags, 1)
    acf_list = []
    for i in lags:
        acf_list.append(auto_correlation(array_to_plot, i))
    plt.stem(lags, acf_list)
    plt.title("ACF Plot for Residuals")
    plt.ylabel("Values")
    plt.xlabel("Lags")
    plt.show()




