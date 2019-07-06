import pandas as pd

#To load the data, we can use Pandas' Dataframe read_table method.
#This allows us to define a separator (in this case, a tab) and
#rename the columns accordingly:

df = pd.read_table('C:/Users/Paolo/Desktop/sklearn_tut/smsspamcollection/SMSSpamCollection',  
                   sep='\t', 
                   header=None,
                   names=['label', 'message'])
print(df.head())
#convert the labels from strings to binary values for our classifier
df['label'] = df.label.map({'ham': 0, 'spam': 1})  
print(df.head())

#convert all characters in the message to lower case:
df['message'] = df.message.map(lambda x: x.lower())  
print(df.head())

#Tokenize the messages into into single words using nltk (Natural Language Toolkit)
#First, we have to import and download the tokenizer from the console:
import nltk  
#nltk.download()

#Apply the tokenization:
df['message'] = df['message'].apply(nltk.word_tokenize)  
print(df.head())

#we will perform some word stemming. The idea of stemming is to normalize our text
#for all variations of words carry the same meaning, regardless of the tense.
#One of the most popular stemming algorithms is the Porter Stemmer:
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])  
print(df.head())

#ransform the data into occurrences, which will be the features
#that we will feed into our model:
from sklearn.feature_extraction.text import CountVectorizer

# This converts the list of words into space-separated strings
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()  
counts = count_vect.fit_transform(df['message'])

#We could leave it as the simple word-count per message, but it is better to use
#Term Frequency Inverse Document Frequency, more known as tf-idf:
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)

#BUILD THE MODEL

#split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size
                                                    =0.1, random_state=69)

#initialise the Navie Bayes Classifier and fit the data
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train, y_train)

#evaluate the model
import numpy as np

predicted = model.predict(X_test)

print(np.mean(predicted == y_test))


#confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, predicted))














