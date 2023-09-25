import re
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests
from requests.auth import HTTPBasicAuth
from urllib.request import urlopen, Request
from dateutil import parser, tz
from datetime import datetime, date, timedelta
import holidays
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from dateutil.relativedelta import relativedelta

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class preprocess():
    def __init__(self,path):
        self.path = path
    def select(self,cond,e):
      con,act = zip(*cond)
      return np.select(con,act,e)
    
    def format_date(self,df):
      df.date = df.date.apply(lambda x: x[x.find('(')+1:].replace(')','') if 'ago' in x else x)
      df.date = df.date.apply(lambda x: x.replace('Published ',''))
      cond = [(df.date.str.contains('ET'),df.date.apply(lambda x: x[:re.search('ET',x).end(0)] if re.search('ET',x) is not None else x)),
              (df.date.str.contains('UTC'),df.date.apply(lambda x: x[:re.search('UTC',x).end(0)] if re.search('UTC',x) is not None else x)),
              (df.date.str.contains('GMT'),df.date.apply(lambda x: x[:re.search('GMT',x).end(0)] if re.search('GMT',x) is not None else x))
              ]
      df.date = self.select(cond,df.date)
      cond = [(~pd.to_datetime(df.date,utc=True,errors='coerce').isna(),pd.to_datetime(df.date,utc=True,errors='coerce').dt.tz_convert('US/Eastern')),
              (df.date.str.contains('ET'),df.date.apply(parser.parse, tzinfos={"ET": tz.gettz("Etc/GMT+4")})),
              (df.date.str.contains('UTC'),df.date.apply(parser.parse, tzinfos={"UTC": tz.tzutc()})),
              (df.date.str.contains('GMT'),df.date.apply(parser.parse, tzinfos={"GMT": tz.tzutc()}) - timedelta(hours=16, minutes=0))
              ]
      df.date = self.select(cond,df.date.apply(parser.parse)) #np.timedelta64('NaT')
      df.date = pd.to_datetime(df.date,utc=True,errors='coerce').dt.tz_convert('US/Eastern')
      return df
    
    def classify_source(self,df):
      for i in ['reuters','bloomberg','investing.com']:
        df[f'source_{i}'] = np.where(df['content'].str.lower().str.find(i)>-1,df['content'].str.lower().str.find(i),np.nan)
      
      t = df[[f'source_{i}' for i in ['reuters','bloomberg','investing.com']]].min(axis=1)
      cond = [(t == df['content'].str.lower().str.find('reuters'),'reuters'),(t == df['content'].str.lower().str.find('bloomberg'),'bloomberg'),
              (t == df['content'].str.lower().str.find('investing.com'),'investing.com'),(df['link'].str.contains('www.reuters.com')==True,'reuters')]
      df['source'] = self.select(cond,None)
      cond = [(t == df['content'].str.lower().str.find('reuters'),df['content'].apply(lambda x: x[x.lower().find('reuters')+len('reuters'):])),
              (t == df['content'].str.lower().str.find('bloomberg'),df['content'].apply(lambda x: x[x.lower().find('bloomberg')+len('bloomberg'):])),
              (t == df['content'].str.lower().str.find('investing.com'),df['content'].apply(lambda x: x[x.lower().find('investing.com')+len('investing.com'):]))]
      df['content'] = self.select(cond,df['content'])
    
      for i in ['reuters','bloomberg','investing.com']:
        df[f'source_{i}'] = np.where(df['title'].str.lower().str.find(f'by {i}')>-1,df['title'].str.lower().str.find(f'by {i}'),np.nan)
      
      t = df[[f'source_{i}' for i in ['reuters','bloomberg','investing.com']]].min(axis=1)
      cond = [(t == df['title'].str.lower().str.find('by reuters'),'reuters'),(t == df['title'].str.lower().str.find('by bloomberg'),'bloomberg'),
              (t == df['title'].str.lower().str.find('by investing.com'),'investing.com')]
      df['source'] = self.select(cond,df['source'])
      cond = [(t == df['title'].str.lower().str.find('by reuters'),df['title'].apply(lambda x: x[:x.lower().find('by reuters')])),
              (t == df['title'].str.lower().str.find('by bloomberg'),df['title'].apply(lambda x: x[:x.lower().find('by bloomberg')])),
              (t == df['title'].str.lower().str.find('by investing.com'),df['title'].apply(lambda x: x[:x.lower().find('by investing.com')]))]
      df['title'] = self.select(cond,df['title'])
      df = df.drop([f'source_{i}' for i in ['reuters','bloomberg','investing.com']], axis=1)
      return df
    
    #Obtain the POS
    def get_wordnet_pos(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            #For special POS, we use 'wordnet.NOUN' to denote it
            return wordnet.NOUN
    
    def text_prepro(self,df):
      X = [word_tokenize(df['title'].tolist()[i])+word_tokenize(df['content'].tolist()[i]) for i in range(len(df))]
      stopword = stopwords.words("english")
      x_cleaned = []
      lemmatizer = WordNetLemmatizer()
      for doc in X:
        doc_nw = [w for w in doc if (w.isalpha()) and (w not in stopword)]
        tagged_doc = pos_tag(doc_nw)
        tagged_doc = [lemmatizer.lemmatize(tag[0].lower(), pos=self.get_wordnet_pos(tag[1])) for tag in tagged_doc]
        x_cleaned += [tagged_doc]
      return pd.Series(x_cleaned,index=df.index)
    
    def combine_database(self,df1):
      try:
          base = pd.read_csv(f'{self.path}news_database.csv')
      except:
          base = pd.DataFrame(columns=['title', 'date', 'content', 'link', 'source', 'Date', 'content_cleaned'])
      df = pd.concat([base,df1],ignore_index=True)
      df = df[[x for x in df.columns if 'unnamed' not in x.lower()]]
      df = df.drop_duplicates(['title','link'],ignore_index=True)
      df.to_csv(f'{self.path}news_database.csv')
      return df
    
    def MapHoliday(self,df,start_date=datetime.strftime(datetime.now() - relativedelta(years=1),'%Y-%m-%d'), end_date=datetime.strftime(datetime.now(),'%Y-%m-%d')):
      # Select holiday from country
      cal = USFederalHolidayCalendar()
      us_holidays = cal.holidays(start=start_date, end=end_date).to_pydatetime()
      weekday = 'Mon Tue Wed Thu Fri'
      bday = CustomBusinessDay(holidays=us_holidays, weekmask=weekday)
      #get previous business day
      df['Date'] = pd.to_datetime(df['date']).apply(lambda x: x - bday if x in us_holidays or x.weekday() in [5,6] else x).dt.strftime('%Y-%m-%d')
      return df
    
    def transform(self,_reu,_inv_com):
      df1 = _inv_com.copy()
      df1 = df1[(~df1['content'].isna())&(~df1['date'].isna())&(~df1['date'].isin(['',' ']))]
      df2 = _reu[(~_reu['date'].isna())&(~_reu['date'].isin(['',' ']))]
      cond = [(df2['date'].str.contains('read'),df2['time']),(df2['date'].str.endswith('UTC'),df2['date'])]
      df2['date'] = self.select(cond,df2['date']+' '+df2['time'])
      df2 = df2.drop(['time'],axis=1)
      comb = pd.concat([df1,df2],ignore_index=True)
      comb = comb.drop_duplicates(['title','content'],ignore_index=True)
      comb = comb.drop([x for x in comb.columns if x.startswith('Unnamed')],axis=1)
      comb['date'] = np.where(comb['date'].isna(),'',comb['date'])
      print("Formating Date of News Release")
      comb = self.format_date(comb)
      print("Classifying Source of News")
      comb = self.classify_source(comb)
      comb = self.MapHoliday(comb, start_date = comb.date.min().year)
      #append to database
      comb['content_cleaned'] = self.text_prepro(comb)
      df = self.combine_database(comb)
      return df