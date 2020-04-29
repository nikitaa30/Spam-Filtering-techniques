# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:38:01 2020

@author: Nikita Sharma
"""
import re
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt 

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

import pandas as pd
dataset = pd.read_csv('spam.csv', encoding='ISO-8859-1');

from autocorrect import spell

data = []

for i in range(dataset.shape[0]):
    sms = dataset.iloc[i, 1]

    sms = re.sub('[^A-Za-z]', ' ', sms)

    sms = sms.lower()

    tokenized_sms = wt(sms)
 
    sms_processed = []
    for word in tokenized_sms:
        if word not in set(stopwords.words('english')):
            sms_processed.append(spell(stemmer.stem(word)))

    sms_text = " ".join(sms_processed)
    data.append(sms_text)

from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(data).toarray()
y = dataset.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

result=[]
data1=[]

data1.append(y_test[2092])
s=""
s+=y_pred[0][0]
s+=y_pred[0][1]
s+=y_pred[0][2]
data1.append(s)
uo=dataset.iloc[2092,1]
data1.append(uo)
result.append(data1)

data1=[]
data1.append(y_test[5051])
s=""
s+=y_pred[1][0]
s+=y_pred[1][1]
s+=y_pred[1][2]
s+=y_pred[1][3]
data1.append(s)
uo=dataset.iloc[5051,1]
data1.append(uo)
result.append(data1)

data1=[]
data1.append(y_test[3999])
s=""
s+=y_pred[4][0]
s+=y_pred[4][1]
s+=y_pred[4][2]
s+=y_pred[4][3]
data1.append(s)
uo=dataset.iloc[3999,1]
data1.append(uo)
result.append(data1)
