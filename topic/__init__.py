from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
import graphviz
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump, load
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd()+'\\topic\\')

import keywordSearch

path = os.getcwd()+'\\topic\\model'

class stock():
    def __init__(self, stock):
        self.stock = stock
        
    def detect(self, df):
      df1 = df.copy()
      dt = load(f'{path}\\{self.stock.lower()}.joblib')
      X = keywordSearch.kwVectorizer(df,self.stock)
      #Display Decision Tree Graph
      """dot_graph = export_graphviz(dt,
                      out_file=None, 
                      feature_names=X.columns,
                      class_names=['others',self.stock],
                      filled=True, rounded=True,
                      special_characters=True)
        tree_graph = graphviz.Source(dot_graph)"""
      predicted_y = np.argmax(dt.predict_proba(X),1)
      df[self.stock] = pd.Series(predicted_y,index=df.index)
      stock_dict = df[['title',self.stock]].set_index(['title']).to_dict()[self.stock]
      df2 = df.copy()
      df2[self.stock] = np.where(df2[self.stock].isna(),df2['title'].apply(lambda x: stock_dict.get(x)),df2[self.stock])
      return df2[self.stock]