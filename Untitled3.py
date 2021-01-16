
# coding: utf-8

# In[1]:


## IMPORTS

# Data Manipulation
import numpy as np
import pandas as pd

# Data dtypes
import json
import re
import string
from pandas.io.json import json_normalize

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Textual data manipulation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


# In[2]:


# Getting data from the file
# Data is in json format and is nested json. 
# We will get the data and convert it into pandas dataframa

with open("C:/Users/hp/Desktop/payload.json", 'r', encoding="utf-8") as json_file:
    json_work = json.load(json_file)
    
df_original = json_normalize(json_work)


# In[3]:


# coppying the data frame for our analysis and model creation

df = df_original


# In[4]:


## Data Understanding and manipulation

df.head()


# In[5]:


# Getting to know the types of variables

df.dtypes


# In[6]:


# Getting to know the shape of the data

df.shape


# In[7]:


# Checking if the data has list objects

df.applymap(lambda x: isinstance(x, list)).all()


# In[8]:


# Changing all variable objects from list to str

df['req.ips'] = [','.join(map(str, l)) for l in df['req.ips']]
df['req.subdomains'] = [','.join(map(str, l)) for l in df['req.subdomains']]


# In[9]:


# Checking the overall data info

df.info()


# In[10]:


# Checking if we have any null values

df.isnull().any()


# In[11]:


# Checking the target variable size and details

df.groupby('isSafe').size()


# In[12]:


# getting percentage for target variable as SAFE_PAYLOAD or UNSAFE_PAYLOAD

safe_payload = len(df[df['isSafe'] == True])
safe_payload_percentage = safe_payload/df.shape[0]*100
unsafe_payload_percentage = 100 - safe_payload_percentage

print("safe_payload = ", safe_payload_percentage)
print("unsafe_payload = ", unsafe_payload_percentage)


# In[13]:


# Understanding the data using describe()

df.describe()


# In[14]:


# As We can see the data have unique = 1 therefore .. all these variables
# can be removed for further analysis therefore removing all these variables

df.drop(columns=df.columns[df.nunique()==1], inplace=True)


# In[15]:


df.head()


# In[16]:


# As the "req.body.note.title" contains only name we can remove the variable

del df['req.body.note.title']


# In[17]:


df.head()


# In[18]:


# As we have text data we will have to use NLP techniques
# As the target variable is Boolean we can one hot encode it into 0, 1 
# 0 ( False ), 1 ( True )

df['isSafe'] = df['isSafe'].astype('int')
df.head()


# In[19]:


## Data Visualization

# Understanding data using pairplot
sns.pairplot(df)


# In[20]:


# Understanding data using countplot
sns.countplot(x='isSafe', data=df)


# In[21]:


## Implementing Machine Learning Model

## Splitting the data into test and train sets

# We will split the data into 70/30 split 

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.33)


# In[22]:


test.head()


# In[23]:


train.head()


# In[24]:


length_train = train['req.body.note.desc'].str.len()
length_test = test['req.body.note.desc'].str.len() 
plt.hist(length_train, label="train_payload") 
plt.hist(length_test, label="test_payload") 
plt.legend() 
plt.show()


# In[25]:


# Cleaning the text data

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and train datasets
train['req.body.note.desc'] = train['req.body.note.desc'].apply(lambda x: clean_text(x))
test['req.body.note.desc'] = test['req.body.note.desc'].apply(lambda x: clean_text(x))

# updated text
train['req.body.note.desc'].head()


# In[26]:


# We will turn the sentence into smaller chuncks using tokenization to remove 
# stopwords and lemmatization techniques

tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
train['req.body.note.desc'] = train['req.body.note.desc'].apply(lambda x:tokenizer.tokenize(x))
test['req.body.note.desc'] = test['req.body.note.desc'].apply(lambda x:tokenizer.tokenize(x))
train['req.body.note.desc'].head()


# In[27]:


# getting all the stopwords

len(stopwords.words('english'))


# In[28]:


# removing stopwords

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words 
train['req.body.note.desc'] = train['req.body.note.desc'].apply(lambda x : remove_stopwords(x))
test['req.body.note.desc'] = test['req.body.note.desc'].apply(lambda x : remove_stopwords(x))
test.head()


# In[29]:


# lemmatization

lem = WordNetLemmatizer()
def lem_word(x):
    return [lem.lemmatize(w) for w in x]


# In[30]:


train['req.body.note.desc'] = train['req.body.note.desc'].apply(lem_word)
test['req.body.note.desc'] = test['req.body.note.desc'].apply(lem_word)


# In[31]:


# Checking data

train['req.body.note.desc'][:10]


# In[32]:


# Combining all the text into one

def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train['req.body.note.desc'] = train['req.body.note.desc'].apply(lambda x : combine_text(x))
test['req.body.note.desc'] = test['req.body.note.desc'].apply(lambda x : combine_text(x))
train['req.body.note.desc']
train.head()


# In[33]:


# Label encoding all the values of the text data using count vectorizer

count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(train['req.body.note.desc'])
test_vector = count_vectorizer.transform(test['req.body.note.desc'])
print(train_vector[0].todense())


# In[34]:


# Implementing Tf-IDF also onto our train data

tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf = tfidf.fit_transform(train['req.body.note.desc'])
test_tfidf = tfidf.transform(test['req.body.note.desc'])


# In[35]:


# As the data is categorical and have 1000 rows we can use KNN to build our model
# Implementing knn and checking the accuracy on the train
# data using cross_val_score

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)

scores_vector = model_selection.cross_val_score(classifier, train_vector, train['req.body.note.desc'], cv = 5, scoring = "f1_micro")
print("score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(classifier, train_tfidf, train['req.body.note.desc'], cv = 5, scoring = "f1_micro")
print("score of tfidf:",scores_tfidf)


# In[36]:


# Predicting the test data using the model

classifier.fit(train_tfidf, train['isSafe'])
y_pred = classifier.predict(test_tfidf)
test['predict'] = y_pred


# In[37]:


# Checking Precision, Recall, and giving the
# classification report for the prediction


print(classification_report(test['isSafe'], test['predict']))
print(confusion_matrix(test['isSafe'], test['predict']))
print(accuracy_score(test['isSafe'], test['predict']))
print("Precision:",metrics.precision_score(test['isSafe'], test['predict']))
print("Recall:",metrics.recall_score(test['isSafe'], test['predict']))


# In[38]:


# As we can see the accuracy is 40% we will try to improve this model . 
# we will try to use logistic regression
# we will use random.seed(), random_state while splitting the data
# we will use stratified K fold in cross_val_score (cv = skf)
# doing the process again to improve our model.


# In[39]:


import random 

random.seed(10)
train1, test1 = train_test_split(df, test_size=0.33, random_state=10)


# In[40]:


# Cleaning the text data

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Applying the cleaning function to both test and train datasets
train1['req.body.note.desc'] = train1['req.body.note.desc'].apply(lambda x: clean_text(x))
test1['req.body.note.desc'] = test1['req.body.note.desc'].apply(lambda x: clean_text(x))


# In[41]:


tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
train1['req.body.note.desc'] = train1['req.body.note.desc'].apply(lambda x:tokenizer.tokenize(x))
test1['req.body.note.desc'] = test1['req.body.note.desc'].apply(lambda x:tokenizer.tokenize(x))


# In[42]:


def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words 
train1['req.body.note.desc'] = train1['req.body.note.desc'].apply(lambda x : remove_stopwords(x))
test1['req.body.note.desc'] = test1['req.body.note.desc'].apply(lambda x : remove_stopwords(x))


# In[43]:


lem = WordNetLemmatizer()
def lem_word(x):
    return [lem.lemmatize(w) for w in x]


# In[44]:


train1['req.body.note.desc'] = train1['req.body.note.desc'].apply(lem_word)
test1['req.body.note.desc'] = test1['req.body.note.desc'].apply(lem_word)


# In[45]:


def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train1['req.body.note.desc'] = train1['req.body.note.desc'].apply(lambda x : combine_text(x))
test1['req.body.note.desc'] = test1['req.body.note.desc'].apply(lambda x : combine_text(x))
train1.head()


# In[46]:


count_vectorizer = CountVectorizer()
train_vector1 = count_vectorizer.fit_transform(train1['req.body.note.desc'])
test_vector1 = count_vectorizer.transform(test1['req.body.note.desc'])
print(train_vector1[0].todense())


# In[47]:


tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf1 = tfidf.fit_transform(train1['req.body.note.desc'])
test_tfidf1 = tfidf.transform(test1['req.body.note.desc'])


# In[48]:


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=2, shuffle=True)


# In[49]:


lg = LogisticRegression(C = 1.0)

scores_vector = model_selection.cross_val_score(lg, train_vector1, train1['req.body.note.desc'], cv = skf, scoring = "f1_micro")
print("score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(lg, train_tfidf1, train1['req.body.note.desc'], cv = skf, scoring = "f1_micro")

print("score of tfidf:",scores_tfidf)


# In[50]:


lg.fit(train_tfidf1, train1['isSafe'])
y_pred1 = lg.predict(test_tfidf1)


# In[51]:


test1['predict'] = y_pred1


# In[52]:


test1.head()


# In[53]:


print(classification_report(test1['isSafe'], test1['predict']))
print(confusion_matrix(test1['isSafe'], test1['predict']))
print(accuracy_score(test1['isSafe'], test1['predict']))
print("Precision:",metrics.precision_score(test1['isSafe'], test1['predict']))
print("Recall:",metrics.recall_score(test1['isSafe'], test1['predict']))


# In[55]:


# Analyzing confusion matrix we get the following measures
# TP = 88, TN= 190, FN= 2, FP=50
# Accuracy = TP + TN / TP + TN + FP + FN = 84.24 %
# Precision = TP / TP + FP = 0.79166
# Recall = TP / TP + FN = 0.9895

####   From Nimra Zafar
####   Submitting from here to Monoxor ---- Thank You for Reviewing

