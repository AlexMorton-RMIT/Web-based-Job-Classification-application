# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, render_template, request, session, redirect
from gensim.models.fasttext import FastText
import pandas as pd
import pickle
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

"""
Note: a faster version of `gen_docVecs`.
"""
def docvecs(embeddings, docs):
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)
        vecs[i,:] = docvec
    return vecs

""" 
stopwords code start 
"""
stopwords = []
with open('static/stopwords_en.txt') as f:
    stopwords = f.read().splitlines()

def remove_stopwords(text_list): 
    new_list =[]
    for text in text_list: 
        if text not in stopwords: 
            new_list.append(text)
    return(new_list)
""" 
stopwords code end 
"""

""" 
Tokenisation Function start
"""
def tokenizeDescs(text):  
    toks = []
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    text = text.lower() # cover all words to lowercase
    tokenizer = nltk.RegexpTokenizer(pattern) 
    tokenised_article = tokenizer.tokenize(text)
    toks.append(tokenised_article)
    return toks
""" 
Tokenisation Function end
"""

""" 
remove repeating tokens start
"""
def remove_repeats(toklist): 
    fixed_list = []
    for t in toklist: 
        if t not in fixed_list: fixed_list.append(t)
    return(fixed_list)
""" 
remove repeating tokens end
"""

""" 
Importing job data set start
"""
dir_path = "static/data/"
folders = [] # list to store the article ID
for foldername in sorted(os.listdir(dir_path)): 
    folders.append(foldername.split(".")[0])

doc_txts = [] # list to store the raw text

for folder_path in folders: 
    path = "static/data/" + folder_path 

    for filename in sorted(os.listdir(path)): 

        current = []
        toSave = []

        if filename.endswith(".txt"):
            curpath = os.path.join(path,filename)
            with open(curpath, "r", encoding="unicode_escape") as f: 
                current.append(f.readlines())
                toSave.append(current[0][0][7:-1])
                toSave.append(current[0][-1][13:-1])
                toSave.append(folder_path)
                doc_txts.append(toSave)
                f.close


jobs = pd.DataFrame(doc_txts, columns=['Title', 'Description', 'category'])
folders = folders[1:]

tokenized_txt = []

for i in range(len(jobs)): 
    to_tokenise = []

    # create tokens list
    to_tokenise.extend((tokenizeDescs(jobs['Description'][i]))[0])
    to_tokenise.extend((tokenizeDescs(jobs['Title'][i]))[0])
    to_tokenise = remove_stopwords(to_tokenise)

    tokenized_txt.append(remove_repeats(to_tokenise))

jobs.insert(3, 'Tokens',tokenized_txt, True)
jobs['Tokens'] = [[lemmatizer.lemmatize(j) for j in jobwords]for jobwords in jobs['Tokens']]
""" 
Importing job data set end
"""
    

app = Flask(__name__)
app.secret_key = os.urandom(16) 

@app.route('/')
def index():
    return render_template('home.html', name="you")

@app.route('/search', methods=['GET', 'POST'])
def about():
    if request.method == 'POST': 

        searched = request.form['searchText']
        searched = searched.lower()

        searchToks = remove_stopwords(tokenizeDescs(searched))

        searchToks = [lemmatizer.lemmatize(w) for w in searchToks[0]] 

        similarities = []

        for job in jobs['Tokens']:             
            simscore = 0
            for tok in searchToks: 
                if tok in job: 
                    simscore = simscore + 1
            similarities.append(simscore)
    
        to_load = []
        
        jobindex = list(jobs.index)



        while max(similarities) > 0: 
            for j in jobindex: 
                if  similarities[j]== max(similarities) : 
                    to_load.append(j)
                    jobindex.remove(j)
                    similarities[j] = 0

        
        return render_template('results.html', jobIndices= to_load, jobTitles = jobs['Title'])
    else: 
        return render_template('search.html')

@app.route('/business/<job_id>')
def business(job_id):
    # Get the job title and description based on job_id
    job_title = jobs['Title'].get(int(job_id), 'Job not found')
    job_description = jobs['Description'].get(int(job_id), 'Description not available')
    
    return render_template('business.html', jobT=job_title, jobD=job_description)

@app.route('/submitted')
def submitted():
    return render_template('submitted.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        f_title = request.form['title']
        f_content = request.form['description']

        tokenized_data = f_content.split(' ')

        # Load the FastText model
        bbcFT = FastText.load("static\desc_FT.model")
        bbcFT_wv= bbcFT.wv

        # Generate vector representation of the tokenized data
        bbcFT_dvs = docvecs(bbcFT_wv, [tokenized_data])

        # Load the LR model
        pkl_filename = "static\descFT_LR.pkl"
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)

        # Predict the label of tokenized_data
        y_pred = model.predict(bbcFT_dvs)
        y_pred = y_pred[0]

        session["title"] = f_title
        session["description"] = f_content
        session["pred"] = y_pred
        return redirect('/confirm')
    else:
        return render_template('classify.html')

@app.route('/confirm', methods=['GET', 'POST'])
def confirm():
    if request.method == 'POST':

        title = session.pop("title", None)
        description = session.pop('description', None)
        category = request.form.get('dropdown')

        newTokens = remove_stopwords(tokenizeDescs(description)[0])
        newTokens.extend(remove_stopwords(tokenizeDescs(title)[0]))
        newTokens = remove_repeats(newTokens)
        newTokens = [lemmatizer.lemmatize(j) for j in newTokens]

        if not title or not description:
            return "Bad Request: Missing title or description in session data", 400
        jobs.loc[len(jobs)]=[title, description, category, newTokens]
        return redirect('/submitted')
    else:
        title = session.get("title")
        description = session.get("description")
        predicted = session.get("pred")
        if not title or not description:
            return redirect('/classify')
        return render_template('confirm.html', title=title, description=description, predicted=predicted, options=folders)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
