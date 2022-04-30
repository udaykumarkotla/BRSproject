from application import app;
from flask import render_template, request, json , Response,redirect,url_for

import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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
    bookName=request.form.get('search')
    algorithm=request.form.get('algorithm')
    if(algorithm=="1"):
        return redirect(url_for('index'))
    elif(algorithm=="2"):
        return redirect(url_for('sameAuthor',bName=bookName))
    elif(algorithm=="3"):
        return redirect(url_for('samePublisher',bName=bookName))

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

    bookName=request.args.get('bName') # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    number=20
    if bookName in list(dataset1['Book-Title'].unique()):
        get_books(dataset1, bookName, number)
    else:
        print("Invalid Book Name!!")
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
    bookName=request.args.get('bName') # Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))
    number=20
    if bookName in list(dataset1['Book-Title'].unique()):
        get_books(dataset1, bookName, number)
    else:
        print("Invalid Book Name!!")
    pb=[]
    for i in range(len(pubBooks)):
        getbook=book_data.loc[book_data['Book-Title'] == pubBooks[i],:]
        getbook.reset_index(inplace=True)
        getbook=book_data.loc[book_data['ISBN'] == getbook['ISBN'][0],:]
        pb.append(getbook)
    return render_template("index.html",Res=pb,str="Books by Same Publisher")

