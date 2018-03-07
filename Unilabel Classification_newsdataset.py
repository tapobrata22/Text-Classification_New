# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:01:59 2018

@author: Tapobrata Behera
"""

from datetime import datetime
import pandas as pd
import re
import time as tt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV 

import os
os.chdir('D:/HACKATHONS/news article freelancing')
os.getcwd()

# grab the data
news = pd.read_csv("uci-news-aggregator.csv")

time1 = datetime.strptime(tt.ctime(), "%a %b %d %H:%M:%S %Y")

def normalize_text(s):
    s = s.lower()    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)    
    return s

news['text'] = [normalize_text(s) for s in news['TITLE']]

news.CATEGORY = np.where(news.CATEGORY == 'b', 'Business', news.CATEGORY)
news.CATEGORY = np.where(news.CATEGORY == 't', 'Tech', news.CATEGORY)
news.CATEGORY = np.where(news.CATEGORY == 'e', 'Entertainment', news.CATEGORY)
news.CATEGORY = np.where(news.CATEGORY == 'm', 'Health', news.CATEGORY)
news.CATEGORY.value_counts()

#CountVectorizer example
from sklearn.feature_extraction.text import CountVectorizer
docs = ["You can catch more flies with honey than you can with vinegar.", "You can lead a horse to water, but you can't make him drink."]
vect = CountVectorizer(min_df=0., max_df=1.0)
X = vect.fit_transform(docs)
print(pd.DataFrame(X.A, columns=vect.get_feature_names()).to_string())

#First implementing Countvecctorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(news['text'])
X_train_counts.shape

'''
docs_new = ['How to become the next Apple or Google of USA', 'Best storage device of europe',"Justin's rocking musics video"]
x_new_tfidf = count_vect.transform(docs_new)
predicted = clf_tuned.predict(x_new_tfidf)
for i in range(0,len(docs_new)):
    print("'",docs_new[i],"'","has been predicted to belong to category -",predicted[i])
'''

#Now doing df_idf transformation
tfidf_transformer = TfidfTransformer()
X_vect_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_vect_tfidf.shape

#Breaking into train and test sample
x_train, x_test, y_train, y_test = train_test_split(X_vect_tfidf, news, test_size=0.25, random_state=42)
y_train22 = y_train.CATEGORY
x_train.shape
y_train22.shape
y_train.CATEGORY.value_counts()
len(y_train)
#Same number of rows for x_train and y_train22 means train and test datasets have been mapped correctly

#Plain MultinomialNB
from sklearn.naive_bayes import MultinomialNB
multi_nb = MultinomialNB().fit(x_train, y_train22)
y_multiNB = multi_nb.predict(x_test)

#Simple SVC 
clf = LinearSVC()
clf.fit(x_train, y_train22)
y_pred = clf.predict(x_test)

#Tuned SVC classifier - 
#keeping C more than 1 
#Dual=False since number of samples(105605) is more than number of features(54637))
#class-weight = balanced since all classes don't have equal occurences
clf_tuned = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=10, multi_class='ovr', 
                      fit_intercept=True, intercept_scaling=1, class_weight="balanced", verbose=0, random_state=None, max_iter=1000)
clf_tuned.fit(x_train, y_train22)
y_pred_svc_tuned = clf_tuned.predict(x_test)

#SGDC classifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])
 
text_clf.fit(x_train, y_train22)
y_pred_sgdc = text_clf.predict(x_test)

x_test.shape
y_test.shape
y_pred.shape
y_pred_sgdc.shape
y_pred_svc_tuned.shape

#Incorporating results of all algorithms
y_test['multiNB'] = y_multiNB
y_test['pred_simple_svc'] = y_pred
y_test['pred_tuned_svc'] = y_pred_svc_tuned
y_test['pred_sgdc'] = y_pred_sgdc
output_test = y_test[['ID','text','CATEGORY','multiNB','pred_simple_svc','pred_tuned_svc','pred_sgdc']]
output_test['multiNB_equality'] = np.where(output_test.CATEGORY == output_test.multiNB, 1, 0)
output_test['simple_svc_equality'] = np.where(output_test.CATEGORY == output_test.pred_simple_svc, 1, 0)
output_test['tuned_svc_equality'] = np.where(output_test.CATEGORY == output_test.pred_tuned_svc, 1, 0)
output_test['sgdc_equality'] = np.where(output_test.CATEGORY == output_test.pred_sgdc, 1, 0)

print("MultiNB accuracy is",(sum(output_test['multiNB_equality'])*100/len(output_test)))
print("Simple SVC accuracy is",(sum(output_test['simple_svc_equality'])*100/len(output_test)))
print("Tuned SVC accuracy is",(sum(output_test['tuned_svc_equality'])*100/len(output_test)))
print("SGDC accuracy is",(sum(output_test['sgdc_equality'])*100/len(output_test)))

#Checking on self made results for tuned SVC
count_vect._validate_vocabulary()

docs_new = ['How to become the next Apple or Google of USA', 'Best storage device of europe',"Justin's rocking musics video"]
x_new_tfidf = count_vect.transform(docs_new)
predicted = clf_tuned.predict(x_new_tfidf)
for i in range(0,len(docs_new)):
    print("'",docs_new[i],"'","has been predicted to belong to category -",predicted[i])

#Checking on self made results for SGDC
docs_new = ['How to become the next Apple or Google of USA', 'Best storage device of europe',"Justin's rocking musics video"]
x_new_tfidf = count_vect.transform(docs_new)
predicted = text_clf.predict(x_new_tfidf)
for i in range(0,len(docs_new)):
    print("'",docs_new[i],"'","has been predicted to belong to category -",predicted[i])
   
#Checking for accuracy

#Printing classification report and confusion matrix
print (classification_report(y_test.CATEGORY, y_pred))
