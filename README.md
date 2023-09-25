# Predict Short Term Stock Trend based on News Sentiment Analysis
Right investment in the stock market might make someone become a millionaire. However, the stock volatility is hard to predict as there might be multiple factors such as politics, economic situations, company’s financial performance etc. This is a project to study what factors might be used in predicting the stock volatility. In recent years, it is easier for people to obtain different kinds of information from the Internet and media. The sentiment of news is one of the information resources that investors study for investment decision making.  Recent research found that news related to infectious diseases will have certain impact to investment strategies among public and thus influencing stock market returns throughout the COVID-19 pandemic (Li, Z. et al., 2022). It indicates that news would be one of the major factors in affecting stock movements. Thus, this project aims at building a prediction model on the stock volatility for investors reference before making the investment decision. To achieve this, this research extract the features and sentiments in terms of financial news in stock prediction to build a prediction model for stock volatility prediction.  

## Extraction and Preprocessing of Online News Articles 

2 online news media will be chosen to collect daily stock news: 1) Reuters, which is reached by billion of people every day (Thomson Reuters, 2022); 2) Investing.com, which has approximately 3 billion monthly pageviews as of July in year 2022 (Investing.com, n.d.). To collect the recent daily news articles, we have parsed the URLs, title and date of each article from the corresponding news websites, followed by crawling content of each article based on the URLs that we have parsed using python package “BeautifulSoup”. After obtaining news content and title, we shall undergo data preprocessing by tokenization, removing stop words and special characters for topic detection and sentiment analysis. 

## Model Design
Several procedures, 1) identify keywords from news content, 2) topic detection, and 3) news sentiment analysis, shall be performed for news data analysis before combining results as illustrated below: 


### Topic Modeling 

After extracting and preprocessing news articles, we will predict the relation of our technology stocks with news articles via topic modeling. This paper shall experiment on decision tree classification model towards topic detection. Meanwhile, complement naive Bayes (CNB) algorithm shall also be experimented in view of imbalanced datasets size. To implement the model, extracted news shall be sampled first to label it manually for training and testing purposes. Then, keyBert model, a training model that can help identify keywords with different N-grams by maximizing the words similarity to the document and minimizing the similarity among words, is used to identify topic key-unigrams for each targeted stock using the labeled news. After identifying the keywords from keyBert model, Decision Tree classifier with Grid Search method while Naïve Bayes Classifier shall be trained and tested respectively for its accuracy and precision to select the optimal topic model for each stock’s sentiment analysis. 

### Sentiment Analysis 

After identifying the potential stocks that shall be affected by the news articles, a sentiment analysis model shall be built to help classify the polarity of the online news based on its title and content. 

Previous researches have performed sentiment analysis to examine the polarity of news articles in different approaches, such as lexicon-based approach, LDA model. Before implementing the model, we shall firstly label the polarity of the news by linking up dates of news release with corresponding stock movement determined by indicator SMA-10 days annotated as -1 (bearish), 0 (neutral/no change), 1 (bullish). For example, when news is released on Day 0, we shall map the corresponding stock’s trend determined on the same date (Day 0), and label the news as Bullish/Bearish after mapping. This paper shall also examine the effect of news in upcoming 5 trading days, so the polarity of news is also labeled for Day 1 to Day 5 respectively. 

 

Illustration of how News is labeled with bullish/bearish for upcoming 5 trading days 

In view of having lengthy sentences to be analyzed and trained, we shall experiment with Bi-LSTM (Bi-directional Long Short Term Memory) models to train the datasets due to the satisfactory results obtained from pioneering studies for instant stock prediction from news content (Sridhar and Sanagavarapu, 2022) (Ren et al., 2020). The LSTM model, a modified recurrent neural network that consists of 4 gates (input gate, forget gate, cell gate, and output gate) to learn long term dependencies in data by preventing vanishing or exploding gradient problems. Bi-LSTM model, a sequence processing model that contains forward and backward direction of LSTM, can help increase the amount of information available to the network and generate meaningful output. Meanwhile, Random Forest Classifier shall also be experimented for comparing with Bi-LSTM model to study the effect of news. To establish the model, grid searching is performed for parameters (dropout, learning rate, activation function) in order to obtain the optimized result. 

## Result
In this study, there were a total of 8289 news collected from our proposed news sources during June to July for the news sentiment analysis as indicated below: 

### Topic Modeling 

Decision Tree Classification has been established with grid search towards 2 parameters (‘max_depth’ and ‘min_sample_leaf’) to obtain the optimized results using gini impurity knowing that it has less computational complexity and faster speed comparing to entropy. Below would be the visualized decision trees for identifying stock’s related news articles based on keywords, which shows that majority of the keywords are either key indicators of corresponding stocks or some hot topics during the period. 

Results showed that the accuracy generally falls approximately 0.9 for topic detection towards training and testing data using decision tree classifier.  

 

Moreover, according to the confusion matrix and the result summary as indicated below, it is shown that the precision mostly falls above 0.75, which indicates that it would have low false positive rate. However, topic detection of news regarding stock ADBE would be lower compared to others as the number of existing ADBE related news are quite few compared to others. 
