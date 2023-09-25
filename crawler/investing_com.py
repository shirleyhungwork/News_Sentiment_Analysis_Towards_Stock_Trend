import os
import sys
sys.path.append(os.getcwd()+'\\crawler\\setting')

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from requests.auth import HTTPBasicAuth
from urllib.request import urlopen, Request
from dataset_conf import DatasetConfiguration

config = DatasetConfiguration()
config.load(os.getcwd()+'\\crawler\\setting\\investing_com.cfg')

session = requests.Session()
auth = HTTPBasicAuth(config.user, config.pw)

def _get_link(url,links):
  html = requests.get(url).content
  soup = BeautifulSoup(html, 'html.parser')
  module = soup.find_all("div", {"class": "largeTitle"})
  print(module)
  for classes in module:
    for link in classes.findAll('a', href=True):
      if link['href'] not in links and 'news' in link['href']:
        links.append(link['href'])
  return links

def get_link(start_page,end_page,*args):
  links = []; start = int(start_page); end = int(end_page)
  for i in range(start,end+1):
    url = config.base_api_url.format(page=str(i))
    try:
      links = _get_link(url,links)
    except:
      print('unable to extract from',url)
  return set(links)

def articles(out_path, start_page=config.start_page,end_page=config.end_page,verbose=0): 
  print('Getting Links from Investing.com ...')
  links = list(get_link(start_page,end_page))
  print('Scrapping Contents from Investing.com ...')
  for i in links:
    print(i)
    if 'http' not in i:
      i = 'https://www.investing.com'+i
    try:
      response = requests.get(i,auth=auth).content
      soup = BeautifulSoup(response,'html.parser')
      content = ''
      try:
        elements = soup.find_all("div", {"class": "WYSIWYG articlePage"})
        for ele in elements:
          for y in ele.find_all('p'):
            content += y.getText()
      except:
        print('Unable to get content from \n',i)
      try:
        title = soup.title.getText()
      except:
        title = ''
      date = ''
      try:
        elements = soup.find_all("div", {"class": "contentSectionDetails"})
        for ele in elements:
          dates = ele.find_all('span')
          for d in dates:
            date += d.getText()
      except:
        date = date
    
      if verbose == 1:
        print('title:',title)
        print('content:',content[:100],'...')
        print('link:',i)
      info += [[title,date,content,i]]
      df = pd.DataFrame(info,columns=['title','date','content','link'])
      df.to_csv(out_path + 'investing_com_articles.csv') #checkpoint
    except:
      info += [['','','',i]]
      if verbose == 1:
        print('Unable to webscrap for link',i)
      df = pd.DataFrame(info,columns=['title','date','content','link'])
      df.to_csv(out_path + 'investing_com_articles.csv') #checkpoint
  return df