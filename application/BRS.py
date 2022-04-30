#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")
#user_data=pd.read_csv('Dataset/Users.csv',header=[0])
book_data=pd.read_csv('static/9A_Project/Dataset/Books.csv',header=[0])
#ratings_data=pd.read_csv('Dataset/Ratings.csv',header=[0])
#book_data.describe()
#book_data.describe()
#ratings_data.describe()


# In[3]:


#print(book_data.head)
#DATA CLEANING

#book_data.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)
#book_data.head()


# In[4]:


book_data.isnull().sum() 


# In[5]:


book_data.loc[book_data['Book-Author'].isnull(),:]


# In[6]:


book_data.loc[book_data['Publisher'].isnull(),:]


# In[7]:


book_data.at[187689 ,'Book-Author'] = 'Other'
book_data.at[128890 ,'Publisher'] = 'Other'
book_data.at[129037 ,'Publisher'] = 'Other'


# In[8]:


book_data['Year-Of-Publication'].unique()


# In[9]:


#pd.set_option('display.max_colwidth', -1)
book_data.loc[book_data['Year-Of-Publication'] == 'DK Publishing Inc',:]


# In[10]:


book_data.loc[book_data['Year-Of-Publication'] == 'Gallimard',:]


# In[11]:


book_data.at[209538 ,'Publisher'] = 'DK Publishing Inc'
book_data.at[209538 ,'Year-Of-Publication'] = 2000
book_data.at[209538 ,'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
book_data.at[209538 ,'Book-Author'] = 'Michael Teitelbaum'

book_data.at[221678 ,'Publisher'] = 'DK Publishing Inc'
book_data.at[221678 ,'Year-Of-Publication'] = 2000
book_data.at[209538 ,'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
book_data.at[209538 ,'Book-Author'] = 'James Buckley'

book_data.at[220731 ,'Publisher'] = 'Gallimard'
book_data.at[220731 ,'Year-Of-Publication'] = '2003'
book_data.at[209538 ,'Book-Title'] = 'Peuple du ciel - Suivi de Les bergers '
book_data.at[209538 ,'Book-Author'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'


# In[12]:


book_data['Year-Of-Publication'] = book_data['Year-Of-Publication'].astype(int)


# In[13]:


book_data.loc[book_data['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2002
book_data.loc[book_data['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002


# In[14]:


book_data['ISBN'] = book_data['ISBN'].str.upper()


# In[15]:


book_data.drop_duplicates(keep='last', inplace=True) 
book_data.reset_index(drop = True, inplace = True)


# In[16]:


#userdataset_preprocessing
users=pd.read_csv('static/9A_Project/Dataset/Users.csv',header=[0])


# In[17]:


## Checking null values
print(users.isna().sum())


# In[18]:


users.drop(['Location'], axis=1, inplace=True)
#print(users.info())


# In[19]:


users.drop_duplicates(keep='last', inplace=True)
users.reset_index(drop=True, inplace=True)
#print(users.info())


# In[20]:


#Ratings dataset cleaning

ratings=pd.read_csv('static/9A_Project/Dataset/Ratings.csv',header=[0])


# In[21]:


print("Columns: ", list(ratings.columns))
#ratings.info()


# In[22]:


ratings.isnull().sum() 


# In[23]:


ratings['ISBN'] = ratings['ISBN'].str.upper()


# In[24]:


ratings.drop_duplicates(keep='last', inplace=True)
ratings.reset_index(drop=True, inplace=True)


# In[25]:


#MERGING DATASETS
dataset = pd.merge(book_data, ratings, on='ISBN', how='inner')
dataset = pd.merge(dataset, users, on='User-ID', how='inner')
#dataset.info()
#dataset.head()


# In[26]:


dataset1 = dataset[dataset['Book-Rating'] != 0]
dataset1 = dataset1.reset_index(drop = True)
#dataset1.info()


# In[27]:


dataset2 = dataset[dataset['Book-Rating'] == 0]
dataset2 = dataset2.reset_index(drop = True)
#dataset2.shape


# In[66]:


#WEIGHTED AVERAGE

df = pd.DataFrame(dataset1['Book-Title'].value_counts())
df['Total-Ratings'] = df['Book-Title']
df['Book-Title'] = df.index
df.reset_index(level=0, inplace=True)
df = df.drop('index',axis=1)
df = pd.read_pickle('weightedData')

## C - Mean vote across the whole
C = df['Average Rating'].mean()

## Minimum number of ratings required to be in the chart
df = df.loc[df['Total-Ratings'] >= 50]
m = df['Total-Ratings'].quantile(0.90)  

def weighted_rating(x, m=m, C=C): 
    v = x['Total-Ratings']    #v - number of votes
    R = x['Average Rating']   #R - Average Rating   
    return (v/(v+m) * R) + (m/(m+v) * C)

df = df.loc[df['Total-Ratings'] >= m]

df['score'] = df.apply(weighted_rating, axis=1)
df = df.sort_values('score', ascending=False)

print("Recommended Books:-\n")
top=df.head(15)
top.reset_index(inplace=True);
print(top)


# In[63]:


getbook=book_data.loc[book_data['Book-Title'] == top['Book-Title'][0],:]
getbook.reset_index(inplace=True);


# In[67]:


getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
print(getbook)


# In[46]:


#Book By Same Author and Same Publisher of given book

def printBook(k, n):
    z = k['Book-Title'].unique()
    for x in range(len(z)):
        print(z[x])
        if x >= n-1:
            break
def get_books(dataframe, name, n):
    print("\nBooks by same Author:\n")
    d = dataframe[dataframe['Book-Title'] == bookName]
    #print(d)
    au = d['Book-Author'].unique()
    print(au)
    data = dataset1[dataset1['Book-Title'] != name]

    if au[0] in list(data['Book-Author'].unique()):
        k2 = data[data['Book-Author'] == au[0]]
    k2 = k2.sort_values(by=['Book-Rating'])
    printBook(k2, n)

    print("\n\nBooks by same Publisher:\n")
    au = d['Publisher'].unique()
    print(au)
    if au[0] in list(data['Publisher'].unique()):
        k2 = pd.DataFrame(data[data['Publisher'] == au[0]])
    k2=k2.sort_values(by=['Book-Rating']) 
    printBook(k2, n)
bookName=input("Enter book name") # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
number=int(input("Enter number of books to be recommended"))

if bookName in list(dataset1['Book-Title'].unique()):
    get_books(dataset1, bookName, number)
else:
    print("Invalid Book Name!!")


# In[71]:


#Collaborative 


#claculates total-ratings each book got and creates a new dataframe having columns bt,isbn,ba,ui,br,tr
df = pd.DataFrame(dataset1['Book-Title'].value_counts())
df['Total-Ratings'] = df['Book-Title']
df['Book-Title'] = df.index
df.reset_index(level=0, inplace=True)
df = df.drop('index',axis=1)
df = dataset1.merge(df, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
df = df.drop(['Year-Of-Publication','Publisher','Age'], axis=1)
#print(df.head(10))
popularity_threshold = 80
popular_book = df[df['Total-Ratings'] >= popularity_threshold]
popular_book = popular_book.reset_index(drop = True)

testdf = pd.DataFrame()
testdf['ISBN'] = popular_book['ISBN']
testdf['Book-Rating'] = popular_book['Book-Rating']
testdf['User-ID'] = popular_book['User-ID']
testdf = testdf[['User-ID','Book-Rating']].groupby(testdf['ISBN'])
#print(testdf.groups) #have the data of users who rated each individual book
#print(dataset1.info())

listOfDictonaries=[]
indexMap = {}
reverseIndexMap = {}
ptr=0

for groupKey in testdf.groups.keys():
    tempDict={}
    groupDF = testdf.get_group(groupKey)
    groupDF= groupDF.reset_index(drop = True)
    for i in range(0,len(groupDF)):
        tempDict[groupDF.iloc[i,0]] = groupDF.iloc[i,1] #[i,0]=user-id [i,1]=rating
    #in tempdict key=userid value=rating
    indexMap[ptr]=groupKey #indexMap has isbn values
    reverseIndexMap[groupKey] = ptr
    ptr=ptr+1
    listOfDictonaries.append(tempDict) 

dictVectorizer = DictVectorizer(sparse=True)
vector = dictVectorizer.fit_transform(listOfDictonaries)
#print(vector)
#pairwiseSimilarity = cosine_similarity(vector)
pairwiseSimilarity = cosine_similarity(vector)

def printBookDetails(bookID):
    print(dataset1[dataset1['ISBN']==bookID]['Book-Title'].values[0])
    

def getTopRecommandations(bookID):
    collaborative = []
    row = reverseIndexMap[bookID]
    print("Input Book:")
    printBookDetails(bookID)
    
    print("\nRECOMMENDATIONS:\n")
    
    mn = 0
    similar = []
    for i in np.argsort(pairwiseSimilarity[row])[::-1]:
          if dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0] not in similar:
               
                if mn>=5:
                      break
                mn+=1
                similar.append(dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0])
                printBookDetails(indexMap[i])
                collaborative.append(dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0])
                #collaborative.append(pairwiseSimilarity[i][np.argsort(pairwiseSimilarity[i])[::-1][0]])
            
    return collaborative

k = list(dataset1['Book-Title'])
m = list(dataset1['ISBN'])
bookName="Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
collaborative = getTopRecommandations(m[k.index(bookName)])
print(collaborative)


# In[69]:


#NEAREST NEIGHBOURS

from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors

data = (dataset1.groupby(by = ['Book-Title'])['Book-Rating'].count().reset_index().
        rename(columns = {'Book-Rating': 'Total-Rating'})[['Book-Title', 'Total-Rating']])

result = pd.merge(data, dataset1, on='Book-Title',)
result = result[result['Total-Rating'] >= popularity_threshold]
result = result.reset_index(drop = True)

matrix = result.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
up_matrix = csr_matrix(matrix)

model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(up_matrix)

bookName="Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
distances, indices = model.kneighbors(matrix.loc[bookName].values.reshape(1, -1), n_neighbors = 5+1)
print("\nRecommended books:\n")
for i in range(0, len(distances.flatten())):
    if i > 0:
        print(matrix.index[indices.flatten()[i]])


# In[ ]:




