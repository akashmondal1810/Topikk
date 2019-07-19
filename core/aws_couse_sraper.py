
# coding: utf-8

# In[1]:


import requests
import urllib.request
from bs4 import BeautifulSoup
from IPython.display import display
import pandas as pd
import json


# In[2]:


fundamental = [
    'unsupervised learning', 'supervised learning', 'semi-supervised learning','neural networks', 
    'learning from observation',
    'dimensionality reduction', 'clustering', 'anamoly detection', 'gan',
    't sne', 'pca', 'autoencoder',
    'mixture model', 'mean shift clustering', 'k means clustering', 'kmeans clustering', 'hierarchical clustering', 
    'expectation maximization', 'density based spatial clustering', 'agglomerative', 
    'divisive', 'elliptic envelope', 'one class svm',
    
    'classification', 'binary classification', 'ensemble learning', 'loss function', 'non optimization based', 'regression',
    'decision tree', 'gradient boost', 'support vector machine', 'svm',
    'bootstrap aggregating bagging', 'boosting', 'ada boost', 'brown boost', 
    'graddient boosting', 'logit boost', 'lp boost', 'random forest', 'stacking',
    'k nn', 'k nearest neighbor', 'decision tree regressor', 'gradient boost regressor',
    'boltzmann machine', 'dbm', 'rbm', 'hebbian learning', 
    'hopfield network', 'multilayer feed forward network', 'mask rcnn', 'rcnn',
    'convolutional neural network', 'cnn','multi layer perceptron', 'radial basis network',
    'perceptrons', 'recurrent neural network', 'rnn', 'lstm network', 'lstm',
    'descision trees', 'descision tree', 'general logical descriptions',
    'inductive learning', 'information theory', 'reinforcement learning',
    'deep learning'
    
]
thematics = [
    'natural language processing', 'nlp', 'perception', 'robotics',
    'discourse', 'semantics', 'speech', 'syntax', 'automatic summarization',
    'coreference resolution', 'lexical semantics', 'machine translation', 
    'named entity recognition', 'natural language generation', 'natural language understanding',
    'ocr', 'question answering', 'relationship extraction', 'sentiment analysis',
    'textual entailment recognition', 'topic segmentation', 'word sense disambiguation',
    'speech recognition', 'speech segmentation', 'text to speech', 'lemmatization',
    'morphological segmentation', 'parsing', 'pos tagging', 'sbd', 'stemming',
    'terminology extraction', 'word segmentation', 'computer vision', '3d vision',
    'digital image processing', 'intermediate level vision', 'low level vision',
    'machine vision', 'neural network based', 'pattern recognition', '3d reconstruction',
    'image rectification', 'invariant and perspective', 'motion', 'optical flow',
    'stereo from motion', 'the kalman filter', 'time to adjacency analysis', 
    'wide baseline matching', 'ahe', 'clahe', 'gcn', 'binary shape analysis', 
    'boundary pattern analysis', 'circle and ellipse detection', 'hough transform',
    'line detection', 'pattern matching', 'basic image filtering', 'edge detection',
    'mathematical morphology', 'texture', 'corner and interest point detection',
    'thresholding', 'adaptive thresholding', 'cnn', 'kernel method', 'statistical',
    'svm', 'effector', 'sensor'
]

learnings = [
    'machine learning', 'deep learning', 'big data', 'data analytics', 'data analysis'
]
learning_list = [fundamental, thematics, learnings]

editor_file_name = 'aws_course_editor.csv'
# In[3]:


def list_desc_norm(text):
    
    import re
    from nltk.corpus import stopwords
    from bs4 import BeautifulSoup
    from nltk.stem import WordNetLemmatizer
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    text = text.strip()
    st = re.compile(r'\W+', re.UNICODE).split(text)
    st = [lemmatizer.lemmatize(str(i)) for i in st if i.lower() not in stop_words]
    
    return ' '.join(str(e.lower()) for e in st)


def get_subject_ai(desc, llist):
    if desc is None:
        return ''
    sub_ai = []
    desc_list = list_desc_norm(desc)
    
    for word in llist[0]:
        if word in desc_list:
            sub = 'Fundamental.'+ str(word)
            sub_ai.append(sub)
    for word in llist[1]:
        if word in desc_list:
            sub = 'thematics.'+ str(word)
            sub_ai.append(sub)
    return ','.join(e for e in sub_ai)

def is_ml_course(desc, llist):
    if desc is None:
        return 'NO'
    is_ml = 'NO'
    desc_list = list_desc_norm(desc)
    
    for word in llist[0]:
        if word in desc_list:
            is_ml = 'YES'
    for word in llist[1]:
        if word in desc_list:
            is_ml = 'YES'
    for word in llist[2]:
        if word in desc_list:
            is_ml = 'YES'
    return is_ml

def lang_norm(lang_dict):
    try:
        rt = lang_dict['Name']
    except:
        rt = ''
    return rt


# In[4]:


def get_scraped_df():
    url = 'https://www.aws.training/api/v1/localizedlearningobject/all'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for el in soup:
        s = el    
    st = str(s)

    x = json.loads(st)
    #print(x[1])


    fields = ['Id', 'Title', 'Description', 'DisplayDuration', 'Language']
    df = pd.DataFrame(columns = fields)

    for data_items in x:
        row = []
        for keys in fields:
            val = data_items[keys]
            row.append(val)
        df.loc[len(df), :] = row
    
    return df




# In[5]:

def get_clean_df():
	df = get_scraped_df()
	df['Language'] = df['Language'].apply(lambda x: lang_norm(x))
	avail_lang = ['English']
	for lng in avail_lang:
	    df = df[df.Language == lng]   #droping val with diffrent language(not in avail_lang)

	df.rename(columns={'DisplayDuration': 'timeReqired'}, inplace=True)
	df['is_ml'] = [is_ml_course(catg, learning_list) for catg in df['Description']]
	df['subject_ai'] = [get_subject_ai(catg, learning_list) for catg in df['Description']]
	df['uri'] = ['https://www.aws.training/learningobject/wbc?id='+str(catg) for catg in df['Id']]

	#	display(df)

	df_fields = [
	    'contributor_author', 'contributor_other', 'type', 'date.created', 'relation_haspart',
	    'institution', 'courseType', 'interactivityType', 'educationalAlignment_educationalLevel',
	    'educationalAlignment_difficultyLevel', 'certificationIfAny', 'isThereAssessment'
	]
	for col in df_fields:
	    df[col] = ""
	df.to_csv(editor_file_name, index=False)


# In[16]:


def add_edit_df():
    import math
    import pandas as pd
    import os.path
    
    if not os.path.exists(editor_file_name):
      print("The File %s it's not created "%editor_file_name)
      print("Generating dataframe " + str(editor_file_name)+'...')
      get_clean_df()
      print("dataframe downloaded")
    
    dafr = pd.read_csv(editor_file_name)
    dafr = dafr.replace({pd.np.nan: None})
    headers = list(dafr.columns.values)
    for index, row in dafr.iterrows():
        for col in headers:
            if row[col] is None:
                input_st = 'pls enter suitable valuse for '+str(col)+ ': '
                row[col] = input(input_st)
            else:
                print(str(col) + ': ' + str(row[col]))
        print('--------------------------*------------------------------')
        dafr.to_csv(editor_file_name, index=False)
        dafr = pd.read_csv(editor_file_name)
        dafr = dafr.replace({pd.np.nan: None})


# In[17]:


add_edit_df()

