from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def decision_tree(filename):
    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )
    print(data)



    vectorizer = TfidfVectorizer(max_features=2500, min_df = 1, max_df = 0.8) #0.8 removes all words that occur in more than 0.8 percent of tweets, which is a LOT
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    clf = DecisionTreeClassifier(criterion="gini", splitter='best')
    print(np.shape(X_tr))
    print(np.shape(y_tr))
    print(np.shape(X_te))
    print(np.shape(y_te))
    print(type(X_te))
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)

    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred))
    print(accuracy_score(y_te, pred))



if __name__=='__main__':
    decision_tree('data_trim_processed.csv')
