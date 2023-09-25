import pandas as pd
import numpy as np
import mpu.io
import json
import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
path = os.getcwd()+'\\topic\\stock_keywords\\'

def kwVectorizer(df,stock):
  df1 = df.copy()
  kw = eval(mpu.io.read(f'{path}{stock.lower()}.json'))
  X = pd.DataFrame()
  for i in kw:
    X[i] = np.where(df1.content_cleaned.apply(lambda x: i in x),1,0)
  return X