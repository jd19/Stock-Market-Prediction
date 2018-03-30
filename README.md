# Stock-Market-Prediction

Goal is predicting the directional stock price change based on the recent financial news and historical price data.
First, we classify a news which related to the target company has a positive or negative effect on the company’s stock price.
Next, use financial indicators and the classified news to identify the historical trends.
Finally, we predict that the stock price goes up or down after this news is released

The process is explained in the flowchart:- 

![capture1](https://user-images.githubusercontent.com/5020590/38136922-78905f6c-33d6-11e8-9675-3847693eb4b6.PNG)

## Feature Extraction

After processing the news articles data features were recorded in a document-term matrix.
For each stock, a separate document term matrix was created using 1-gram features.
The initial dictionary that we used, had a large number of words in it. It had 85,131 words in it which makes up a huge feature space and therefore reduce it
We use stemming to achieve this result

## News Labeling
News collected from various sources were labeled so that they could be used for classification.
The idea is that news had an impact on the directionality of stock prices.
Initially we labeled the news using the directions of the stock prices in the same hour. 
That is, if the news was at 10 am, then the open and close values at that time were used to label the news.
Labelling is done as follows :

Label = 1 if Close < Open
Label = -1 otherwise

## Classification

Once the news articles of all the companies are labelled based on the directional change of stocks we trained different classifiers and recorded their accuracies.
Several classifiers like logistic regression,KNN and neural networks are used.
