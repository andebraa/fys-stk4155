from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#TODO:
#stemmer!
#alter stopwords??
#bigger dataset?


def random_forest(filename):
    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )
    print(data)



    vectorizer = TfidfVectorizer( min_df = 1, max_df = 0.8)
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    clf = RandomForestClassifier(n_estimators = 100, criterion="gini", n_jobs=-1)
    print(np.shape(X_tr))
    print(np.shape(y_tr))
    print(np.shape(X_te))
    print(np.shape(y_te))
    print(type(X_te))
    clf.fit(X_tr, y_tr)

    pred_te = clf.predict(X_te)
    pred_train = clf.predict(X_tr)
    print("-----------------------------------")
    print(confusion_matrix(y_te, pred_te))
    print(classification_report(y_te, pred_te))
    print(accuracy_score(y_te, pred_te))
    print("-----------------------------------")
    print(confusion_matrix(y_tr, pred_train))
    print(classification_report(y_tr, pred_train))
    print(accuracy_score(y_tr, pred_train))



if __name__=='__main__':
    random_forest('data_trim_processed.csv')
