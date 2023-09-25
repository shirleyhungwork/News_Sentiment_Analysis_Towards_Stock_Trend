# Predict Short Term Stock Trend based on News Sentiment Analysis
Right investment in the stock market might make someone become a millionaire. However, the stock volatility is hard to predict as there might be multiple factors such as politics, economic situations, company’s financial performance etc. This is a project to study what factors might be used in predicting the stock volatility. In recent years, it is easier for people to obtain different kinds of information from the Internet and media. The sentiment of news is one of the information resources that investors study for investment decision making.  Recent research found that news related to infectious diseases will have certain impact to investment strategies among public and thus influencing stock market returns throughout the COVID-19 pandemic (Li, Z. et al., 2022). It indicates that news would be one of the major factors in affecting stock movements. Thus, this project aims at building a prediction model on the stock volatility for investors reference before making the investment decision. To achieve this, this research extract the features and sentiments in terms of financial news in stock prediction to build a prediction model for stock volatility prediction.  

## Extraction and Preprocessing of Online News Articles 

2 online news media will be chosen to collect daily stock news: 1) Reuters, which is reached by billion of people every day (Thomson Reuters, 2022); 2) Investing.com, which has approximately 3 billion monthly pageviews as of July in year 2022 (Investing.com, n.d.). To collect the recent daily news articles, we have parsed the URLs, title and date of each article from the corresponding news websites, followed by crawling content of each article based on the URLs that we have parsed using python package “BeautifulSoup”. After obtaining news content and title, we shall undergo data preprocessing by tokenization, removing stop words and special characters for topic detection and sentiment analysis. To successfully execute the web-scraping task, it is required to input your user name and password in files "investing_com.cfg" and "reuters.cfg" inside folder "crawler" -> "setting" in order to link request for API

## Model Design
Several procedures, 1) identify keywords from news content, 2) topic detection, and 3) news sentiment analysis, shall be performed for news data analysis before combining results as illustrated below: 

### Topic Modeling 

After extracting and preprocessing news articles, we will predict the relation of our technology stocks with news articles via topic modeling. This paper shall experiment on decision tree classification model towards topic detection. Meanwhile, complement naive Bayes (CNB) algorithm shall also be experimented in view of imbalanced datasets size. To implement the model, extracted news shall be sampled first to label it manually for training and testing purposes. Then, keyBert model, a training model that can help identify keywords with different N-grams by maximizing the words similarity to the document and minimizing the similarity among words, is used to identify topic key-unigrams for each targeted stock using the labeled news. After identifying the keywords from keyBert model, Decision Tree classifier with Grid Search method while Naïve Bayes Classifier shall be trained and tested respectively for its accuracy and precision to select the optimal topic model for each stock’s sentiment analysis. 

### Sentiment Analysis 

After identifying the potential stocks that shall be affected by the news articles, a sentiment analysis model shall be built to help classify the polarity of the online news based on its title and content. 

Previous researches have performed sentiment analysis to examine the polarity of news articles in different approaches, such as lexicon-based approach, LDA model. Before implementing the model, we shall firstly label the polarity of the news by linking up dates of news release with corresponding stock movement determined by indicator SMA-10 days annotated as -1 (bearish), 0 (neutral/no change), 1 (bullish). For example, when news is released on Day 0, we shall map the corresponding stock’s trend determined on the same date (Day 0), and label the news as Bullish/Bearish after mapping. This paper shall also examine the effect of news in upcoming 5 trading days, so the polarity of news is also labeled for Day 1 to Day 5 respectively. 
<p align="center">
  <img src="/img/model_design.jpg" alt="Image" width="600"> 
</p>
<p align="center">Illustration of how News is labeled with bullish/bearish for upcoming 5 trading days</p>

In view of having lengthy sentences to be analyzed and trained, we shall experiment with Bi-LSTM (Bi-directional Long Short Term Memory) models to train the datasets due to the satisfactory results obtained from pioneering studies for instant stock prediction from news content (Sridhar and Sanagavarapu, 2022) (Ren et al., 2020). The LSTM model, a modified recurrent neural network that consists of 4 gates (input gate, forget gate, cell gate, and output gate) to learn long term dependencies in data by preventing vanishing or exploding gradient problems. Bi-LSTM model, a sequence processing model that contains forward and backward direction of LSTM, can help increase the amount of information available to the network and generate meaningful output. Meanwhile, Random Forest Classifier shall also be experimented for comparing with Bi-LSTM model to study the effect of news. To establish the model, grid searching is performed for parameters (dropout, learning rate, activation function) in order to obtain the optimized result. 

## Result
In this study, there were a total of 8289 news collected from our proposed news sources during June to July for the news sentiment analysis as indicated below: 

### Topic Modeling 
Decision Tree Classification has been established with grid search towards 2 parameters (‘max_depth’ and ‘min_sample_leaf’) to obtain the optimized results using gini impurity knowing that it has less computational complexity and faster speed comparing to entropy. Below would be one of the visualized decision trees for identifying Tesla stock’s related news articles based on keywords, which shows that majority of the keywords are either key indicators of corresponding stocks or some hot topics during the period. 
<p align="center">
  <img src="/img/tsla_topic_model.jpg" alt="Image" width="600"> 
</p>

Results showed that the accuracy generally falls approximately 0.9 for topic detection towards training and testing data using decision tree classifier.  
<p align="center">
  <img src="/img/tsla_topic_model_metric.jpg" alt="Image" width="600"> 
</p>

Moreover, according to the confusion matrix and the result summary as indicated below, it is shown that the precision mostly falls above 0.75, which indicates that it would have low false positive rate. However, topic detection of news regarding stock ADBE would be lower compared to others as the number of existing ADBE related news are quite few compared to others. 
<p align="center">
  <img src="/img/tsla_topic_model_metric_2.jpg" alt="Image" width="600"> 
</p>

## Sentiment Analysis 
Bidirectional LSTM model, LSTM model and Random Forest Classifier were experimented based on the predicted results conducted by topic modelling algorithm concluded above (Decision Tree Classification) to correlate news release and trending of stock price in upcoming 5 trading days, in which 80% of detected stock-related datasets were trained whereas the remaining 20% were used for validation. According to the training and validation results as indicated below, it is shown that the validation accuracy generally falls between 0.7 and 0.8 using Bi-LSTM model, while the testing accuracy of LSTM and Random Forest Classifier are approximately 0.6. However, it appears that the effect of stock prediction using Bi-LSTM model towards the ADBE movement is significantly less than others with prediction accuracy approximately 0.69 on average.  
<p align="center">
  <img src="/img/sentiment_metric_1.jpg" alt="Image" width="600"> 
</p>

<p align="center">
  <img src="/img/sentiment_metric_2.jpg" alt="Image" width="600"> 
</p>

<p align="center">
  <img src="/img/sentiment_metric_4.jpg" alt="Image" width="600"> 
</p>

In terms of precision, recall and f1-score, the performance of Bidirectional LSTM model generally would have better performance than LSTM and Random Forest Model with precision, recall, and f1-score around 0.74, 0.74, 0.71 respectively. In addition, it could be indicated that the Bi-LSTM model would have higher efficiency in identifying “True Positive” and “True Negative” while avoiding errors caused by “False Positive” and “False Negative”. Hence, the Bidirectional LSTM model is implemented for predicting stocks’ movement intraday and upcoming 5 days for the corresponding 5 stocks. 

After selecting optimal sentiment analysis model towards stock price prediction, the predicted probability of stock uptrend is aggregated based on number of news content intraday and the overall average testing accuracy among trading days between June and July is shown in Table 4.3.7. According to the result below, it is shown that the accuracy generally falls between 0.8 and 0.9, yet the testing accuracy on Day 0 TSLA stock prediction is comparatively low among others with accuracy approximately 0.72. 
<p align="center">
  <img src="/img/sentiment_metric_5.jpg" alt="Image" width="600"> 
</p>

## Discussion and Conclusion
After identifying keywords among news articles using keyBert model, Decision Tree Classifier and Naïve Bayes classifier have been experimented for topic modelling in which performance of topic detection using decision tree classifier generally better than Naïve Bayes classifier. In addition, it shows satisfactory results in identifying related stock topic using Decision Tree Classification after evaluating the accuracy, precision and F1-score with an accuracy achieving 0.9. However, it would be prone in predicting stock being less under spotlight or having few news related to it. For instance, there was limited news mentioning Adobe Inc. (“ADBE”) during the research period that most stock movements were related to macroeconomic (I.e. economic recession due to inflation policy decided by US Federal Reserve). 

For news sentiment analysis related to stock movements, process has been streamlined by directly correlating news content with stock price for trend classification. Bidirectional LSTM model, LSTM model and Random Forest Classifier were experimented in predicting upcoming 5 days’ stock movements, in which accuracy was mostly an average of 0.84 for Bidirectional LSTM model while about 0.6 for Random Forest Classifier and LSTM model. 

However, the model would have tendency of overfitting due to limited number of news collected during time period and performance is subject to selecting optimal topic detection model. In future work, it is planned to improve the efficiency of news sentiment by creating live update towards existing built model (I.e. decision tree classifier) based on the newly posted news to obtain a more sustainable prediction model and more comprehensive study on the prediction effectiveness between streamlined stock sentiment model and stock prediction based on news polarity. In addition, acquiring larger database for model training might further increase the performance of stock prediction. 
