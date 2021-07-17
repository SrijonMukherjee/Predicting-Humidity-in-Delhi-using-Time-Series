# Predicting Humidity in Delhi using Time Series Analysis
The purpose of this project was to predict the humidity in Delhi from climate variables such as  mean pressure, speed, and temperature (recorded daily from 2013 to 2017). The data set was obtained from [Kaggle](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data) and has 1575 rows. Different models such as Holt Winter's Method, Multivariate Regression, and ARMA models were used on the training and test sets and their results were compared to select the best performing model.

## File Description:

| File    | Description    | 
| ------------- |:-------------:| 
| config.py  |Contains the code for all the functions and methods used in the Project.|
|Predicting Humidity in Delhi.py| Main script containing code for analysis and plots made.|
|daily_climate_delhi.csv| Data for humidity and other climate variables recorded daily in Delhi. Obtained  from [Kaggle](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data).|
|Predicting Humidity in Delhi(Report).pdf|Detailed report containing project description, methods, results and conclusions.|
|Predicting Humidity in Delhi.pptx| PowerPoint Presentation of the project.|

## Instructions to Run:
1. Download data from this repository or download data from [here](https://www.kaggle.com/sumanthvrao/daily-climate-time-series-data).
2. Make sure the data set (CSV file) is in the same directory as the two python scripts.
3. Open config.py and run the whole file.
4. Open Predicting Humidity in Delhi.py and run the script. It will import everything from config.py and output the different models and analysis.

