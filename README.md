# Stock Price Prediction for Companies working in the Biotechnology Industry
## Problem Definition
This project aims to compare the performance of various supervised learning approaches to stock market prediction. In this project, both regression as well as classification algorithms are applied. Limiting the boundaries of the project, only stock market data of companies which are active in the biotechnology field are taken as input.

## Data Sources
In order to perform any data-related task, the initial step is to acquire the data. Firstly, a dataset containing all tickers and information of companies in the U.S. stock market was found on Kaggle. The original dataset can be accessed by clicking on the link below:
[Original Dataset](https://www.kaggle.com/datasets/marketahead/all-us-stocks-tickers-company-info-logos)
These companies are categorized by multiple factors, one of which is their industry. There are 491 indices whose industry is recorded as biotechnology. Therefore, the names and tickers of those were extracted and saved into a separate .csv file.
Next, stock market data of the companies was retrieved from reliable sources, namely Yahoo Finance, IEX Cloud, and Quandl. 118 companies do not have any data recorded which can be accessed for free, thus these companies’ data are yet to be found.
