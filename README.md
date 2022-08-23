# Stock Price Prediction for Companies working in the Biotechnology Industry
## 1 Introduction
This project is the final project of my bachelor’s degree in Computer Engineering, done under supervision of
Dr. Nastooh Taheri Javan. As a sophomore student, I was introduced to the exciting and rapidly-growing
field of Machine Learning. Ever since then, I have been studying the fundamentals of Python programming,
Machine Learning, Statistics, and Mathematics hoping to gain deep knowledge. A few of my study materials
and courses I used during my learning process include Machine Learning Specialization by Stanford University
and DeepLearning.ai, Intro to Machine Learning by Kaggle, Intermediate Machine Learning by Kaggle, and
Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition. Applying the knowledge
I had gained, I chose this project and have been working on it for some time.

In the following sections, a detailed explanation of each step of the process has been written. In addition, the
code can be found here.

## 2 Problem Definition
This project aims to compare the performance of various supervised learning approaches to stock market
prediction. In this project, both classification, as well as regression algorithms are applied. Limiting the
boundaries of the project, only stock market data of companies active in the biotechnology field are taken as
input. The target variable is the Close feature, which specifies the closing price of a stock. While in the case of
the classification algorithms, the purpose is to predict whether the Close feature increases or decreases on the
following day, for the regression algorithms the motive is to forecast the exact closing price.

## 3 Data Collection
To perform any data-related task, the initial step is to acquire the data. Firstly, a data set containing all tickers
and information of companies in the U.S. stock market was found on Kaggle. The original data set can be
accessed by clicking on the link below: [View Initial Data set](https://www.kaggle.com/datasets/marketahead/all-us-stocks-tickers-company-info-logos) on Kaggle

These companies are categorized by multiple factors, one of which is their industry. There are 491 indices
whose industry is recorded as biotechnology. Therefore, the names and tickers of those were extracted and saved
into a separate .csv file. Next, stock market data of the previously extracted companies were retrieved from
reliable sources, namely Yahoo Finance and IEX Cloud. Unfortunately, 119 of 491 companies do not have any
data recorded available for free, thus these companies’ data are yet to be found. The retrieved data is stored as
.csv files to be used as input for the rest of the project.
All notebooks relating to this section are accessible through the link below:

[Data Collection Notebooks](https://github.com/shbnmzr/stock-market-prediction/tree/main/1-data-collection)

## 4 Returns Analysis
Returns Analysis consists of analyzing each company’s accumulative return, daily return, comparing their
percent return throughout the period, Sharpe ratio, Sortino ratio, and Probabilistic Sharpe Ratio.

### 4.1 Initial Exploration
Initial exploration is done in order to study all the features and their impact on the target or prediction.
As was seen in the code, our data is a Time series, whose index is of type DatetimeIndex, and columns are
of type float. Additionally, the most number of observations recorded on a single company is 1258, although
not all of them have as many indices.

### 4.2 Examining Correlations
Correlations among the features needed to be studied. Therefore, a heatmap indicating the correlations was
plotted. In all illustrations, a high correlation among 4 features, Close, Open, High, and Low was seen. In most
cases, another feature, Adjacent Close, also revealed a high correlation with the Close feature.

This visualization confirms the initial suspicion that four features are highly correlated, and thus should be
eliminated, since it may result in ”Multicollinearity”. Multicollinearity can lead to misleading results in some
algorithms. At the end of this phase, there are 3 features remaining, Close, Adj Close, and Volume. The reason
Adj Close will not be dropped at this point is that it will be used in some computations later on.

All photos showing the correlations among each company’s features are stored [here](https://github.com/shbnmzr/stock-market-prediction/tree/main/photos/correlations).

### 4.3 Daily and Accumulative Returns Analysis
Firstly, each company’s accumulative return was computed and a line diagram was generated and stored. 

All the plots can be viewed [here](https://github.com/shbnmzr/stock-market-prediction/tree/main/photos/accumulative-returns).

Secondly, each company’s daily return throughout the time period was computed and a respective line diagram was generated and stored. The companies’ daily returns were diverse.
All the plots can be viewed [here](https://github.com/shbnmzr/stock-market-prediction/tree/main/photos/daily-returns). Next, the percentage change of all companies was computed. The result of
this phase was also surprising. Although some indices demonstrated great positive changes, there were some
others that showed highly negative numbers.

### 4.4 Sharpe Ratio
The Sharpe ratio can be used to evaluate the total performance of an aggregate investment portfolio or the
performance of an individual stock. The Sharpe ratio indicates how well an equity investment performs in
comparison to the rate of return on a risk-free investment, such as U.S. government treasury bonds or bills.

### 4.5 Sortino Ratio
In contrast to Sharpe ratio, Sortino ratio only punishes returns falling below a user-defined threshold. Additionally,
instead of inspecting the complete standard deviation, Sortino ratio only computes the downside risk,
i.e the standard deviation of all returns which are smaller than the target.


### 4.6 Probabilistic Sharpe Ratio
The problem of Sharpe ratio is, that it is calculated via historical data, and thus it only yields an estimation
and not the true Sharpe ratio.

All notebooks relating to this phase are available through the link below:

[Returns Analysis Notebooks](https://github.com/shbnmzr/stock-market-prediction/blob/main/2-exploratory-data-analysis/Returns_Analysis.ipynb)

## 5 Preprocessing
This phase is done to prepare the data for the next phase, which is modelling. The implementation of this step
is [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/3-preprocessing/Feature_Engineering.ipynb).

### 5.1 Lag Features
To make a lag feature, the observations of the target series are shifted so that they appear to have occurred later
in time. In this project, 5 lag features are created. Lag features let us model serial dependence. A time series
has serial dependence when an observation can be predicted from previous observations, which is the primary
goal.

### 5.2 Other Features
In addition to lag features, I assumed other factors affect stock market prices, as well. Therefore, not only
lag features but also Volume, Daily Return, Monthly Moving Average, and Quarterly Moving Average were
also added to the Dataframes before being input to models. This step resulted in tremendous performance
improvement.

### 5.3 Feature Scaling
#### 5.3.1 Z-Score Normalization
This step is done in order to prepare our data for the next phases. In this project, Z-score Normalization
is performed on the Close feature in order to make the values comparable.

#### 5.3.2 Min Max Normalization
Another way of normalizing the values is to apply Min-max normalization.

## 6 Models
Simply put, a Time series is a set of observations collected over time. In this project, data from 21-08-2017 to
18-08-2022 has been recorded.

Lag Feature → to make a lag feature we shift the observations of the target series so that they appear to have
occurred later in time.

### 6.1 Classification
#### 6.1.1 Logistic Regression
Logistic Regression estimates the probability of that an instance belongs to a particular class, making it a Binary
Classification algorithm. If the estimated probability is greater than a threshold (usually 0.5), then the model
predicts that the instance belongs to the Positive Class, otherwise it is classified as belonging to the Negative
Class.

This model computes a weighted sum of the input features, X, plus a bias term, b, then outputs the logistic of
this result. The logistic is a sigmoid function which outputs a number between 0 and 1. The cost function over the whole training set is the average cost over all training examples.

To optimize the algorithm, Gradient Descent is utilized, which tweaks the parameters attractively to minimize
the cost function. 

The Logistic regression model applied in this project can be accessed through this [link](https://github.com/shbnmzr/stock-market-prediction/blob/main/4-models/classification/Logistic_Regression.ipynb).

#### 6.1.2 Gradient Boosting
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble. It begins
by initializing the ensemble with a single model, whose predictions can be pretty naive. The cycle begins:
- The current ensemble is used to generate predictions for each observation in the data set. Adding the
predictions from all models in the ensemble, a prediction is made.
- These predictions are used to calculate a loss function.
- As done in Logistic Regression, Gradient Descent is used to optimize the parameters.
- A new model is added to the ensemble.
- These steps are repeated!

XGBoostClassifier model has been implemented and is available [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/4-models/classification/XGBoost.ipynb).

#### 6.1.3 Neural Networks
A sequential neural network model is used to predict whether the close price inclines or declines the next day.
Here, an 8-layered neural network is constructed, the first 7 layers(Input and Hidden layers) of which have
”relu” activation functions and the final layer has a ”linear” activation unit. The number of units in each layer
is 70, 60, 50, 40, 30, 20, 10, and 2, respectively.

The final layer of the neural network generates 2 outputs. One output is selected as the predicted answer. In
the output layer, a vector z is generated by a linear function which is fed into a softmax function. The softmax
function converts z into a probability distribution as described below. After applying softmax, each output will
be between 0 and 1 and the outputs will sum to 1. They can be interpreted as probabilities. The larger inputs
to the softmax will correspond to larger output probabilities. 

The implementation of this model is accessible [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/4-models/classification/Neural_Networks.ipynb).

### 6.2 Regression
#### 6.2.1 Gradient Boosting
Extreme Gradient Boosting algorithm, which has already been explained the previous section, operates the
same way for regression problems. Therefore, I will not repeat the details once more.
XGBoostRegressor model has been implemented and is available [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/4-models/regression/XGBoost.ipynb).

#### 6.2.2 Neural Networks
Same as before, a sequential neural network model is used to predict whether the close price, however, in this
case, our incentive is to predict the exact close price of each stock. Here, an 10-layered neural network is
constructed, the first 9 layers(Input and Hidden layers) of which have ”relu” activation functions and the final
layer has a ”linear” activation unit. The number of units in each layer is 90, 80, 70, 60, 50, 40, 30, 20, 10, and
1, respectively. The final layer containing a single linear unit outputs the prediction. The related notebook can
be viewed [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/4-models/regression/Neural_Networks.ipynb).

## 7 Model Evaluation
To Evaluate the classification models, a confusion matrix is generated for each company’s data. Additionally,
Precision, Recall, and F1 Score are also computed.
Evaluation of regression models, on the other hand, cannot be done using the means explained in the last
paragraph. Here, Mean Squared Error is calculated, a common method for regression models.

## 8 Results
To make this report concise, all results are saved in separate files, accessible [here](https://github.com/shbnmzr/stock-market-prediction/blob/main/reports).