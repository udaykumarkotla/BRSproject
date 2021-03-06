from application import app;
from flask import render_template, request, json , Response,redirect,url_for

import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from pandas.api.types import is_numeric_dtype
from sklearn.neighbors import NearestNeighbors

book_data=pd.read_csv('static/9A_Project/Dataset/Books.csv',header=[0])

book_data.isnull().sum() 

book_data.loc[book_data['Book-Author'].isnull(),:]

book_data.loc[book_data['Publisher'].isnull(),:]

book_data.at[187689 ,'Book-Author'] = 'Other'
book_data.at[128890 ,'Publisher'] = 'Other'
book_data.at[129037 ,'Publisher'] = 'Other'

book_data['Year-Of-Publication'].unique()

book_data.loc[book_data['Year-Of-Publication'] == 'DK Publishing Inc',:]

book_data.loc[book_data['Year-Of-Publication'] == 'Gallimard',:]

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

book_data['Year-Of-Publication'] = book_data['Year-Of-Publication'].astype(int)

book_data.loc[book_data['Year-Of-Publication'] > 2021, 'Year-Of-Publication'] = 2002
book_data.loc[book_data['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = 2002

book_data['ISBN'] = book_data['ISBN'].str.upper()

book_data.drop_duplicates(keep='last', inplace=True) 
book_data.reset_index(drop = True, inplace = True)

users=pd.read_csv('static/9A_Project/Dataset/Users.csv',header=[0])

print(users.isna().sum())

users.drop(['Location'], axis=1, inplace=True)

users.drop_duplicates(keep='last', inplace=True)
users.reset_index(drop=True, inplace=True)

ratings=pd.read_csv('static/9A_Project/Dataset/Ratings.csv',header=[0])

print("Columns: ", list(ratings.columns))

ratings.isnull().sum() 

ratings['ISBN'] = ratings['ISBN'].str.upper()
ratings.drop_duplicates(keep='last', inplace=True)
ratings.reset_index(drop=True, inplace=True)
dataset = pd.merge(book_data, ratings, on='ISBN', how='inner')
dataset = pd.merge(dataset, users, on='User-ID', how='inner')
dataset1 = dataset[dataset['Book-Rating'] != 0]
dataset1 = dataset1.reset_index(drop = True)
dataset2 = dataset[dataset['Book-Rating'] == 0]
dataset2 = dataset2.reset_index(drop = True)
@app.route("/")
@app.route("/index")
def index():
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
    top=df.head(20)
    top.reset_index(inplace=True);
    #print(top)
    res=[]
    for i in range(len(top)):
        getbook=book_data.loc[book_data['Book-Title'] == top['Book-Title'][i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        res.append(getbook)
    #print (res)
    return render_template("index.html",str="TOP RATED BOOKS",Res=res)

@app.route("/searchbook",methods=["GET","POST"])
def searchbook():
    bookName=request.form.get('search').strip()
    algorithm=request.form.get('algorithm')
    if(algorithm=="1"):
        return redirect(url_for('index'))
    elif(algorithm=="2"):
        return redirect(url_for('sameAuthor',bName=bookName))
    elif(algorithm=="3"):
        return redirect(url_for('samePublisher',bName=bookName))
        #Collaborative
    elif(algorithm=="4"):
        return redirect(url_for('collaborative',bName=bookName))
    elif(algorithm=="5"):
        return redirect(url_for('nearestNeighbour',bName=bookName))

@app.route("/sameAuthor")
def sameAuthor():
    authorbooks=[]
    def printBook(arr,k, n):
        
        z = k['Book-Title'].unique()
        for x in range(len(z)):
            arr.append(z[x])
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
        printBook(authorbooks,k2, n)

    bookName=request.args.get('bName').strip() # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    number=20
    authorbooks.append(bookName)
    if bookName in list(dataset1['Book-Title'].unique()):
        get_books(dataset1, bookName, number)
    else:
        return render_template("error.html")
    ab=[]
    for i in range(len(authorbooks)):
        getbook=book_data.loc[book_data['Book-Title'] == authorbooks[i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        ab.append(getbook)
    return render_template("index.html",Res=ab,str="Books by Same Author")

@app.route("/samePublisher")
def samePublisher():
    pubBooks=[]
    def printBook(arr,k, n):
        
        z = k['Book-Title'].unique()
        for x in range(len(z)):
            arr.append(z[x])
            if x >= n-1:
                break
    def get_books(dataframe, name, n):
        d = dataframe[dataframe['Book-Title'] == bookName]
        print("\n\nBooks by same Publisher:\n")
        au = d['Publisher'].unique()
        print(au)
        data = dataset1[dataset1['Book-Title'] != name]
        if au[0] in list(data['Publisher'].unique()):
            k2 = pd.DataFrame(data[data['Publisher'] == au[0]])
        k2=k2.sort_values(by=['Book-Rating']) 
        printBook(pubBooks,k2, n)
    bookName=request.args.get('bName').strip() # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    number=20
    pubBooks.append(bookName)
    if bookName in list(dataset1['Book-Title'].unique()):
        get_books(dataset1, bookName, number)
    else:
        return render_template("error.html")
    pb=[]
    for i in range(len(pubBooks)):
        getbook=book_data.loc[book_data['Book-Title'] == pubBooks[i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        pb.append(getbook)
    return render_template("index.html",Res=pb,str="Books by Same Publisher")

@app.route("/collaborative")
def collaborative():
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
        res = []
        row = reverseIndexMap[bookID]
        print("Input Book:")
        printBookDetails(bookID)
        
        print("\nRECOMMENDATIONS:\n")
        
        mn = 0
        similar = []
        for i in np.argsort(pairwiseSimilarity[row])[::-1]:
            if dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0] not in similar:
                
                    if mn>=20:
                        break
                    mn+=1
                    similar.append(dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0])
                    printBookDetails(indexMap[i])
                    res.append(dataset1[dataset1['ISBN']==indexMap[i]]['Book-Title'].values[0])
                    #collaborative.append(pairwiseSimilarity[i][np.argsort(pairwiseSimilarity[i])[::-1][0]])
                
        return res

    k = list(dataset1['Book-Title'])
    m = list(dataset1['ISBN'])
    bookName=request.args.get('bName').strip()
    if (not (bookName in list(dataset1['Book-Title'].unique()))):
        return render_template("error.html")
    res = getTopRecommandations(m[k.index(bookName)])
    ans=[]
    for i in range(len(res)):
        getbook=book_data.loc[book_data['Book-Title'] == res[i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        ans.append(getbook)
    
    return render_template("index.html",Res=ans,str="Collaborative filtering Technique")

@app.route("/nearestNeighbour")
def nearestNeighbour():
    data = (dataset1.groupby(by = ['Book-Title'])['Book-Rating'].count().reset_index().rename(columns = {'Book-Rating': 'Total-Rating'})[['Book-Title', 'Total-Rating']])
    popularity_threshold = 80

    result = pd.merge(data, dataset1, on='Book-Title',)
    result = result[result['Total-Rating'] >= popularity_threshold]
    result = result.reset_index(drop = True)

    matrix = result.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
    up_matrix = csr_matrix(matrix)

    model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model.fit(up_matrix)

    bookName=request.args.get('bName').strip() #"Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
    if (not (bookName in list(dataset1['Book-Title'].unique()))):
        return render_template("error.html")
    distances, indices = model.kneighbors(matrix.loc[bookName].values.reshape(1, -1), n_neighbors = 15+1)
    print("\nRecommended books:\n")
    res=[bookName]
    for i in range(0, len(distances.flatten())):
        if i > 0:
            res.append(matrix.index[indices.flatten()[i]]) 
    
    ans=[]
    for i in range(len(res)):
        getbook=book_data.loc[book_data['Book-Title'] == res[i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        ans.append(getbook)
    return render_template("index.html",Res=ans,str="Nearest Neighbour Technique")
    
@app.route("/bookDetails",methods=["GET","POST"])
def bookDetails():
    bName=request.form.get('bname')
    au=request.form.get('author')
    pb=request.form.get('publisher')
    yop=request.form.get('yop')
    img=request.form.get('bimage')
    return render_template("bookdetail.html",bname=bName,au=au,pb=pb,yop=yop,img=img,str="Book Details")