# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:13:07 2018

@author: Tapobrata Behera
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os
os.chdir('D:/HACKATHONS/news article freelancing')
os.getcwd()

df = pd.read_csv("movies_genres.csv", delimiter='\t')
df_genres = df.drop(['plot', 'title'], axis=1)
counts = []
categories = list(df_genres.columns.values)

#Checking number of movies per category
for i in categories:
    counts.append((i, df_genres[i].sum()))
df_stats = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats.plot(x='genre', y='#movies', kind='bar', legend=False, grid=True, figsize=(15, 8))

#Since the Lifestyle has 0 instances we can just remove it from the data set
df.drop('Lifestyle', axis=1, inplace=True)
data_df = df

#Converting the text part of dataset first into a vectorsiser and then doing the tf_idf transformation of the text vectoriser. 
#Vectoriser basically converts a column of text into multiple columns of unique words and for each entry we see which word exists in that entry. 
#Tf-Idf transformer then finds the weightage of the words according to their presence in the various entries. 
def text_transformation(dataset, target):
    count_vect = CountVectorizer(stop_words='english')
    X_train_counts = count_vect.fit_transform(dataset[target])
    tfidf_transformer = TfidfTransformer()
    X_vect_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    return X_vect_tfidf

#Function to split the count vectorised and tf_idf transformed dataset to training and testing groups.     
def split_dataset(dataset1, dataset2, train_percentage,seed):
    x_train, x_test, y_train, y_test = train_test_split(dataset1, dataset2, train_size=train_percentage,random_state=seed)
    y_train22 = y_train.drop(['title', 'plot'], axis=1)
    return x_train, x_test, y_train, y_test, y_train22

#Function to run OnevsRest Linear Classifier on the categories of the train data. 
#Inputs are the count_vectorised trained dataframes and catgories of the train dataframe along with the count_vectorised test dataframe
#Output is the predicted categories. 
def SVC_pipeline(train_set1,train_set2,test_set1):
    pipeline_simple_svc = Pipeline([
        (('clf', OneVsRestClassifier(LinearSVC()))),])
    pipeline_simple_svc.fit(train_set1,train_set2)
    pred_simple_svc = pipeline_simple_svc.predict(test_set1)
    return pred_simple_svc

#Function to run Simple Random Forest Classifier on the categories of the train data. 
#Inputs are the count_vectorised trained dataframes and catgories of the train dataframe along with the count_vectorised test dataframe
#Output is the predicted categories. 
def RF_pipeline(train_set1,train_set2,test_set1):
    pipeline_simple_rf = Pipeline([
    (('clf', RandomForestClassifier())),])
    pipeline_simple_rf.fit(train_set1,train_set2)
    pred_simple_rf = pipeline_simple_rf.predict(test_set1)
    return pred_simple_rf
    
#Function to run Tuned Random Forest Classifier on the categories of the train data. 
#Inputs are the count_vectorised trained dataframes and catgories of the train dataframe along with the count_vectorised test dataframe
#Output is the predicted categories. 
    
def tuned_RF_pipeline(train_set1,train_set2,test_set1):
    pipeline_simple_rf = Pipeline([
    (('clf', RandomForestClassifier(n_estimators=50,criterion="gini", max_depth=None, min_samples_split=2, 
                                    min_samples_leaf=50, min_weight_fraction_leaf=0.0, max_features=0.5))),])
    pipeline_simple_rf.fit(train_set1,train_set2)
    pred_simple_rf = pipeline_simple_rf.predict(test_set1)
    return pred_simple_rf

#Function to check accuracy. The test file and the predicted files go into input and the output is the accuracy %.    
def accuracy(test_file, predicted_file):
    y_test_simple_svc = test_file.drop(['title', 'plot'], axis=1)
    acc = accuracy_score(y_test_simple_svc, predicted_file)*100
    return acc

headers = list(data_df)

#Calling function to transform the text(first column) into count_vectoriser and then to tf_idf 
dataset = text_transformation(data_df, headers[1])

#Calling function to break the data into train and &  test sets
x_train, x_test, y_train, y_test, y_train22 = split_dataset(dataset, data_df, 0.75,42)

#Performing the simple SVC classification
y_pred_simple_svc = SVC_pipeline(x_train, y_train22,x_test)

#Performing the simple RF classification
y_pred_simple_rf = RF_pipeline(x_train, y_train22,x_test)

#Performing the simple RF classification
y_pred_tuned_rf = tuned_RF_pipeline(x_train, y_train22,x_test)

#Printing accuracies of models
print("Accuracy of Simple Linear SVM is",accuracy(y_test,y_pred_simple_svc),"%")
print("Accuracy of Simple Random Forest is",accuracy(y_test,y_pred_simple_rf),"%")
print("Accuracy of Tuned Random Forest is",accuracy(y_test,y_pred_tuned_rf),"%")

#Predicting from a self made dataframe
data1 =  {
'Movie' : ["A","B","C"],
'Review' : ["Set in Delhi, the film revolves around lovers who take the whole duration of the film to get married with the blessings of their parents. The technique used to introduce the hero, Veer and heroine, Geet harks back to the 1990s. While the camera concentrates on Veer swagger, in case of Geet, it pans from her backside to her bosom",
"Highly qualified and once wealthy Kailash struggles to make ends meet after hooligans burn down his goods factory owing to growing tension between two dominant religious groups.Set in 1948, the film tries to reiterate the principles of late Mahatma Gandhi while parallely narrating the tale of one man who stands at the opposite spectrum of non-violence. After a chance encounter with a total stranger (Subrat Dutta), while on his way to mend broken ties with his estranged mother, Kailash shows signs of developing a liking towards the path of inclusion and communal harmony.",
"Director Chakri Toleti in his Hindi directorial debut takes viewers behind the scenes of a popular awards show and tries to elicit laughter by laying bare the drama that is an intrinsic aspect of showbiz.Teji (Diljith Dosanjh) and Jinal (Sonakshi Sinha) set out to live their dreams when they get an opportunity to be part of a Bollywood awards night and showcase their talent. But, little do they know that the awards show manager, Sophie (Lara Dutta), has chosen them from among millions of contest entrants, despite being terrible at what they do, only to embarrass her boss, Gary (Boman Irani), and wreak revenge on him for not giving her the due. "]
}

rev = pd.DataFrame(data1,columns = ['Movie','Review'])

#Function to predict results of self-made reviews taking Simple Linear SVC as the classifier:- 
def vect(df,original_df,x_df,y_df):
    new_names = list(df)
    old_names = list(original_df)
    c_vect = CountVectorizer(stop_words='english')
    c_vect.fit_transform(original_df[old_names[1]])
    trans_count_vect = c_vect.transform(df[new_names[1]])
    result_df = SVC_pipeline(x_df, y_df,trans_count_vect)
    print("a")
    result_df = pd.DataFrame(result_df)
    print("b")
    result_df.columns = y_df.columns.tolist()
    return result_df
    
self_predicted_simple_svc_df = vect(rev, data_df,x_train, y_train22)
self_predicted_simple_svc_df.shape

#Interpretation - 
#Four inputs go into this function - The self_made dataframe(named rev), the original dataframe(named data_df), the initially created train dataframe(named x_train) and its corresponding categories. (named y_train22)
#We need the x_train and y_train22 for the SVC pipeline. 
#Resultant dataframe contains the 27 categories as columns and their one-hot encoded variables. 
#Since we entered 3 reviews, we got a 3X27 dataframe.     

 