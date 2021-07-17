from config import *
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chi2

#Loading the data from my dataset
delhi_climate = pd.read_csv("daily_climate_delhi.csv")

#Displaying head:
delhi_climate.head(10)

#Displaying tail:
delhi_climate.tail(10)

#Checking for NAN values:
delhi_climate.info()

#3.a Plot of the dependent variable versus time
plt.plot(delhi_climate.humidity)
plt.ylabel("Humidity")
plt.xlabel("Days")
plt.title("Plot of Humidity versus Days")
plt.show()

#3.b ACF of the dependent variable:
acf_plot(80,delhi_climate.humidity)


#3.c Correlation Matrix with SNS

corr = delhi_climate.corr()
print(corr)
sns.heatmap(corr, annot = True)


#3.e Splitting data set into training and test data sets

X = delhi_climate[['meantemp', 'wind_speed', 'meanpressure']]
Y = delhi_climate['humidity']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, shuffle = False)



#4. Stationarity:
#ADF Stationarity Test


adf_test(delhi_climate.humidity)

# Decomposing the data with additive model

results_additive = seasonal_decompose(delhi_climate.humidity, model = "additive", freq = 1)
results_additive.plot()

plt.show()

#Decomposing data with multiplicative model:
results_multiplicative = seasonal_decompose(delhi_climate.humidity, model = "multiplicative", freq = 1)
results_multiplicative.plot()

plt.show()

#6 Holt-Winters Method
holt_winters = ets.ExponentialSmoothing(Y_train, trend= None, seasonal='add', seasonal_periods=365).fit()
holt_winters_forecast = holt_winters.predict(start = 1260, end = 1574)

#Displaying the forecasts
display(holt_winters_forecast)



#Residuals for Holt Winters
residuals_holts = Y_test - holt_winters_forecast

holts_res_mean = np.mean(residuals_holts)
holts_res_mean
print("The mean of the residuals for Holts Winter is: {}".format(holts_res_mean))
#ACF of Holts Winter forecasts

acf_plot_residuals(120, residuals_holts)


#Variance of Residuals:
print("The variance of the residuals is: {}".format(np.var(residuals_holts)))
holts_var = np.var(residuals_holts)

#RMSE for Holts Winter
holts_rmse = np.sqrt(np.mean(residuals_holts ** 2 ))
holts_rmse
print("The RMSE of the Holts-Winter model is: {}".format(holts_rmse))


#Plotting Holts Winter Forecasts
plt.plot(Y_train, label = 'Train Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(holt_winters_forecast, label = 'Holts Winter Forecast')
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.title("Holts Winter Forecast Results")
plt.legend()
plt.show()






#8 Multiple Linear Regression Model

X_train_fit = sm.add_constant(X_train)
multiple_regression_first_fit = sm.OLS(Y_train, X_train_fit).fit()
multiple_regression_first_fit.summary()

#Removing the meanpressure feature and running another fit:

X_removed_feature = X_train_fit.drop(['meanpressure'], axis = 1)
multiple_regression_second_fit = sm.OLS(Y_train, X_removed_feature).fit()
multiple_regression_second_fit.summary()


#Predictions on OLS model
X_test_fit = sm.add_constant(X_test).drop(['meanpressure'], axis = 1)


y_hat_OLS = multiple_regression_second_fit.predict(X_test_fit)


#Plotting the forecasts
plt.plot(Y_train, label = 'Training Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(y_hat_OLS, label = 'Forecasts')
plt.legend()
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.title("Multiple Regression Forecast Results")
plt.show()


#Residuals

residuals_OLS = Y_test - y_hat_OLS


#RMSE of Residuals from OLS

np.sqrt(np.mean(residuals_OLS ** 2))
ols_rmse = np.sqrt(np.mean(residuals_OLS ** 2))
#ACF of Residuals from OLS model

acf_plot_residuals(120, list(residuals_OLS))

#Q value of Residuals:
print("The Q value of the residuals is: {}".format(find_Q_value(Y_test, y_hat_OLS, 15)))

#Variance of Residuals
print("The variance of the residuals is: {}".format(np.var(residuals_OLS)))

ols_var = np.var(residuals_OLS)
#Mean of OLS Residuals
print("The mean of the residuals is: {}" .format(np.mean(residuals_OLS)))
ols_res_mean = np.mean(residuals_OLS)


#ARMA Model Determination

ry = theoretical_acf(50, Y_train)
ry1 = ry[::-1]
ry2 = np.concatenate((np.reshape(ry1,50), ry[1:]))

#AutoCorrelation plot for Theoretical ACF
plt.stem(ry2)
plt.show()

#gpac table
sns.heatmap(gpac(ry2,7,7), annot = True)


#Trying ARMA(1,0):
#Trying order with na = 1, nb =0:
#ARMA Parameter Estimation:

model_arma_10 = arma_model_parameter_estimation(Y_train,1,0)


#Confidence Intervals for ARMA(1,0):
model_arma_10.conf_int()

#Estimated Covariance:
model_arma_10.cov_params()

#Prediction
y_hat_arma10 = model_arma_10.predict(start = 1260, end  = 1574)

residuals_arma10 = Y_test - y_hat_arma10

#Plot of forecast
plt.plot(Y_train,label = 'Train Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(y_hat_arma10, label = 'Forecast')
plt.title("ARMA (1,0) Model Results")
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.legend()
plt.show()


#Estimated Variance:
print("The estimated variance of the residuals for ARMA(1,0) model is: {}".format(np.var(Y_test - y_hat_arma10)))
arma10_var = np.var(residuals_arma10)

#Mean of Residuals:
arma10_mean = np.mean(residuals_arma10)
arma10_mean
print("Mean of residuals for ARMA(1,0) model: {}".format(arma10_mean))


#ACF
acf_plot_residuals(120, residuals_arma10)


#Chi_Square Test
Q10 = find_Q_value(Y_test, y_hat_arma10, 20)
Q10
dof = 20 - 1 - 0
alpha = 0.01
if Q10 < chi2.ppf(0.99, dof):
   print("Residuals are white")
else:
   print("Residuals are not white")



residuals_arma10 = Y_test - y_hat_arma10

#RMSE of ARMA(1,0):
np.sqrt(np.mean(residuals_arma10 ** 2))

arma10_rmse = np.sqrt(np.mean(residuals_arma10 ** 2))
#Trying ARMA(2,5):
#Trying order with na = 2, nb =5:
#ARMA Parameter Estimation:

model_arma_25 = arma_model_parameter_estimation(Y_train,2,5)

#Confidence intervals ARMA(1,5):
model_arma_25.conf_int()

#Covariance parameters
model_arma_25.cov_params()

#Forecasts
y_hat_arma25 = model_arma_25.predict(start = 1260, end  = 1574)

residuals_arma25 = Y_test - y_hat_arma25


#Plot of forecast
plt.plot(Y_train,label = 'Train Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(y_hat_arma25, label = 'Forecast')
plt.title("ARMA (2,5) Model Results")
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.legend()
plt.show()

#Mean of Residuals:
arma25_mean = np.mean(residuals_arma25)
arma25_mean
print("The residual mean of the ARMA(2,5) model is: {}".format(arma25_mean))
#Estimated Variance

arma25_var = np.var(residuals_arma25)
arma25_var
print("Estimated variance of residuals of ARMA (2,5) is: {}".format(arma25_var))

#ACF
acf_plot_residuals(120, residuals_arma25)


#RMSE residuals
arma25_rmse = np.sqrt(np.mean(residuals_arma25 ** 2))
arma25_rmse
print("The RMSE of the residuals: {}".format(arma25_rmse))

#Chi_Square Test
Q25 = find_Q_value(Y_test, y_hat_arma25, 20)
dof = 20 - 2 - 5
alpha = 0.01
if Q25 < chi2.ppf(0.99, dof):
   print("Residuals are white")
else:
   print("Residuals are not white")


#Trying after subtracting mean (ARMA(1,0)):
Y_subtracted_mean = np.subtract(Y_train, np.mean(Y_train))

model_mean_10 = arma_model_parameter_estimation(Y_subtracted_mean,1,0)

#confidence intervals:
model_mean_10.conf_int()

#forecasts:
y_mean_10 = model_mean_10.predict(start = 1260, end = 1574)
y_mean_10 = np.add(y_mean_10, np.mean(Y_train))

plt.plot(Y_train,label = 'Train Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(y_mean_10, label = 'Forecast')
plt.title("Results of Subtracted Mean Predictions")
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.legend()
plt.show()


#Chi Square
#Chi_Square Test
Q_mean_10 = find_Q_value(Y_test, y_mean_10, 20)
dof = 20 - 1 - 0
alpha = 0.01
if Q_mean_10 < chi2.ppf(0.99, dof):
   print("Residuals are white")
else:
   print("Residuals are not white")


#Trying after subtracting mean (ARMA(2,5)):

model_mean_25 = arma_model_parameter_estimation(Y_subtracted_mean, 2, 5)

#Confidence Intervals
model_mean_25.conf_int()

#forecasts:
y_mean_25 = model_mean_25.predict(start = 1260, end = 1574)
y_mean_25 = np.add(y_mean_25, np.mean(Y_train))

plt.plot(Y_train,label = 'Train Data')
plt.plot(Y_test, label = 'Test Data')
plt.plot(y_mean_25, label = 'Forecast')
plt.title("Results of Subtracted Mean Predictions")
plt.xlabel("Days")
plt.ylabel("Humidity")
plt.legend()
plt.show()

#Chi_Square Test
Q_mean_25 = find_Q_value(Y_test, y_mean_25, 20)
dof = 20 - 1 - 0
alpha = 0.01
if Q_mean_25 < chi2.ppf(0.99, dof):
   print("Residuals are white")
else:
   print("Residuals are not white")





Table = pd.DataFrame(data = {
  'Residual Mean': [holts_res_mean,ols_res_mean,arma10_mean,arma25_mean],
  'Residual Variance': [holts_var,ols_var,arma10_var,arma25_var],
  'RMSE':[holts_rmse,ols_rmse,arma10_rmse,arma25_rmse]



}, index = ['Holts Winter', 'Multiple Regression', 'ARMA (1,0)', 'ARMA(2,5)']
)

display(Table)