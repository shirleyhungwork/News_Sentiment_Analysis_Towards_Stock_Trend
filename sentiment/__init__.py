import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Concatenate, Input, Flatten
from keras import models
from keras.metrics import Precision, Recall
from tensorflow.keras.utils import plot_model

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier

import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os
import sys
sys.path.append(f'{os.getcwd()}\\sentiment\\model')
sys.path.append(f'{os.getcwd()}\\sentiment\\')
sys.path.append(f'{os.getcwd()}\\crawler\\')
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from dateutil.relativedelta import relativedelta

import yf_stock
import time
from datetime import datetime
import mpu
import re

"""
To install below package, you need to download by following commands:
from spacy.cli import download
print(download('en_core_web_sm'))
"""
import en_core_web_sm

path = os.getcwd() + '\\sentiment\\model\\'

nlp = en_core_web_sm.load()
lemmatizer = WordNetLemmatizer()
FILTERS = '!"#$%&()*+,-/:;?@[\]^_`{|}~tn'
DISABLE_PIPELINES = ["tok2vec", "parser", "ner", "textcat", "custom", "lemmatizer"]

def _MapHoliday(s, day,start_date=datetime.strftime(datetime.now() - relativedelta(years=1),'%Y-%m-%d'), end_date=datetime.strftime(datetime.now(),'%Y-%m-%d')):
  # Select holiday from country
  cal = USFederalHolidayCalendar()
  us_holidays = cal.holidays(start=start_date, end=end_date).to_pydatetime()
  weekday = 'Mon Tue Wed Thu Fri'
  bday = CustomBusinessDay(holidays=us_holidays, weekmask=weekday)
  #get previous business day
  return pd.to_datetime(s).apply(lambda x: x + bday*day if x in us_holidays or x.weekday() in [5,6] else x).dt.strftime('%Y-%m-%d')

def NewsOutput(inp,stock,target,day):
  base = yf_stock.extract_stock_price(stock).reset_index()
  base['Date'] = base['Date'].dt.strftime('%Y-%m-%d')
  base = base[base['Date']>='2022-06-01'][['Date']].reset_index(drop=True)
  for i in range(day,6):
    inp[f'next_{i}_day'] = _MapHoliday(inp['Date'],i-day)
    if f'predicted_{stock}_day_{i}_{target}' not in inp.columns:
        inp[f'predicted_{stock}_day_{i}_{target}'] = np.nan
  new = inp[['Date']+[f'next_{i}_day' for i in range(day,6)]+[f'predicted_{stock}_day_{i}_{target}' for i in range(day,6)]]
  new = new.sort_values(['Date']).groupby(['Date']+[f'next_{i}_day' for i in range(day,6)]).mean().reset_index()
  for d in range(day,6):
    base = base.merge(new[[f'next_{d}_day',f'predicted_{stock}_day_{d}_{target}']].rename({f'next_{d}_day':'Date',f'predicted_{stock}_day_{d}_{target}':f'previous_{d}_day_predicted_{target}'},axis=1),how='left',on=['Date'])
  return base

def initial_preprocessing(text):
  """
      - Remove HTML tags
      - For numberings like 1st, 2nd
      - Remove extra characters > 2 eg:
      ohhhh to ohh
  """
  HTML_TAG_PATTERN = re.compile(r']*>')
  tag_removed_text = HTML_TAG_PATTERN.sub('', text)
  NUMBERING_PATTERN = re.compile('d+(?:st|[nr]d|th)')
  numberings_removed_text =  NUMBERING_PATTERN.sub('', tag_removed_text)
  extra_chars_removed_text = re.sub(r"(.)1{2,}",  r'11', numberings_removed_text)
  return extra_chars_removed_text

def preprocess_text(doc):
  """
      Removes the 
      1. Spaces
      2. Email
      3. URLs
      4. Stopwords
      5. Punctuations
      6. Numbers
  """
  tokens = [token for token in doc 
                if not token.is_space and
                not token.like_email and 
                not token.like_url and 
                not token.is_stop and 
                not token.is_punct and 
                not token.like_num]

  """
      Remove special characters in tokens except dot
      (would be useful for acronym handling)
  """
  translation_table = str.maketrans('', '', FILTERS)
  translated_tokens = [token.text.lower().translate(translation_table) for token in tokens]

  """
  Remove integers if any after removing special characters, remove single characters and lemmatize
  """
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in translated_tokens if len(token) > 1]

  return lemmatized_tokens

def bilstm(vocab_size,max_sequence_len,dropout=0.1,alpha=0.01):
  model = Sequential()
  model.add(Embedding(vocab_size, 16, input_length=max_sequence_len))
  model.add(Bidirectional(LSTM(16, input_shape=(None, 1))))
  model.add(Dropout(dropout))
  model.add(Dense(1, activation='sigmoid'))
  adam = Adam(learning_rate=alpha)
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=adam, metrics=['accuracy'], run_eagerly=True)
  return model

def one_hot_encoding(df,kw):
  df1 = df.copy()
  X = df1[['date']]
  X['date'] = pd.to_datetime(X['date'],utc=True).dt.strftime('%Y-%m-%d')
  for i in kw:
    X[i] = np.where(df1.content_cleaned.apply(lambda x: i in x),1,0)
  return X

class news_sentiment:
  def __init__(self,df,stock,target):
    self.df = df
    self.stock = stock.lower()
    self.target = target
    if target == 'Price':
      self.StockPrice = yf_stock.extract_stock_price(self.stock).reset_index()
    self.StockMove = self.linkStock()
    self.vocab_size = None
    self.max_sequence_len = None
    self.token = None
    self.padding = 'post'

  def linkStock(self):
    #stock price
    st_trend = self.StockPrice
    st_trend['Date'] = st_trend['Date'].dt.strftime('%Y-%m-%d')
    df1 = self.df.rename({'AdjDate':'Date'},axis=1).merge(st_trend,how='left',on=['Date'])
    out = df1[(df1['Date']>='2022-05-24')&(df1['content'].notnull())].reset_index(drop=True)
    out['content'] = out['content'].apply(initial_preprocessing)
    return out

  def TestText(self,day):
    """Preprocess the text data"""
    predict = self.StockMove[(self.StockMove[self.stock]==1)]
    texts = [preprocess_text(doc) for doc in nlp.pipe(predict['content'], disable=DISABLE_PIPELINES)]
    sequences = []
    tokenizer = Tokenizer(filters=FILTERS,lower=True)
    tokenizer.fit_on_texts(texts)
    self.token = tokenizer
    for text in texts:
        # convert texts to sequence
        txt_to_seq = self.token.texts_to_sequences([text])[0]
        sequences.append(txt_to_seq)
        # find max_sequence_len for padding
        txt_to_seq_len = len(txt_to_seq)
    # post padding
    model = tf.keras.models.load_model(f'{path}{self.stock}_day_{day}_Price_trend.h5')
    self.max_sequence_len = model.get_config()['layers'][0]['config']['batch_input_shape'][1]
    padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_len, padding=self.padding)
    return padded_sequences

  def predict(self,day):
    model = tf.keras.models.load_model(f'{path}{self.stock}_day_{day}_Price_trend.h5')
    predict = self.StockMove[(self.StockMove[self.stock]==1)].reset_index(drop=True)
    padded_sequences = self.TestText(day)
    predict[f'predicted_{self.stock}_day_{day}'] = pd.Series(np.argmax(model.predict(padded_sequences),1),index=predict.index)
    return predict