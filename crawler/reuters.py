import os
import sys
sys.path.append(os.getcwd()+'\\crawler\\setting')

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from requests.auth import HTTPBasicAuth
from dataset_conf import DatasetConfiguration

config = DatasetConfiguration()
config.load(os.getcwd()+'\\crawler\\setting\\reuters.cfg')

session = requests.Session()
auth = HTTPBasicAuth(config.user, config.pw)

def _get_link(url,links):
  html = requests.get(url).content
  soup = BeautifulSoup(html, 'lxml')
  module = soup.find('div',class_='news-headline-list')
  elements = module.find_all('a')
  for element in elements:
    if 'http' in element['href']:
      links.append(element['href'])
    else:
      links.append('https://www.reuters.com'+element['href'])
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
  print('Getting Links from Reuters...')
  links = list(get_link(start_page,end_page))
  info = []
  print('Scrapping Contents from Reuters...')
  for i in links:
    response = requests.get(i,auth=auth).content
    soup = BeautifulSoup(response,'lxml')
    elements = soup.find_all('p')
    content = ''
    for element in elements:
      try:
        if 'paragraph' in element['data-testid']:
          content += element.getText()
      except:
        regex = re.compile('.*paragraph-.*')
        try:
          if 'paragraph' in element['class'][0].lower():
            content += element.getText()
        except:
          print('Unable to get content from \n',i)
    try:
      title = soup.title.getText()
    except:
      title = ''
    regex = re.compile('.*date-.*')
    try:
      date = soup.find_all('span',{"class" : regex})[0].getText()
    except:
      try:
        regex = re.compile('.*date.*')
        date = soup.find_all('time',{"class" : regex})[0].getText()
      except:
        date = ''
    try:
      time = soup.find_all('span',{"class" : regex})[1].getText()
    except:
      try:
        regex = re.compile('.*date.*')
        time = soup.find_all('time',{"class" : regex})[1].getText()
      except:
        time = ''
    info += [[title,date,time,content,i]]
    if verbose == 1:
      print('title:',title)
      print('content:',content[:100],'...')
      print('link:',i)
    df = pd.DataFrame(info,columns=['title','date','time','content','link'])
    df.to_csv(out_path+'reuter_articles.csv') #checkpoint
  return df