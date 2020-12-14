from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from playsound import playsound


def decision_tree(filename):
    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )
    print(data)

    # data = pd.read_csv(filename, #note; this is for testing purposes
    #     usecols = [0,5],
    #     names=['label', 'tweet'],
    #     encoding='latin1'
    # )



    vectorizer = TfidfVectorizer() #0.8 removes all words that occur in more than 0.8 percent of tweets, which is a LOT
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    clf = DecisionTreeClassifier(criterion="gini", splitter='best', max_depth=100)


    clf.fit(X_tr, y_tr)


    pred_train = clf.predict(X_tr)
    pred_test = clf.predict(X_te)

    # calculate accuracy score
    test_acc = accuracy_score(y_te, pred_test)
    train_acc = accuracy_score(y_tr , pred_train)
    print(test_acc)
    print(train_acc)

    sounds = ['sounds/Not_Gay_Sex.mp3', 'sounds/Objection_Heresay.mp3','sounds/Rock_Flag_and_Eagle.mp3', 'sounds/The_good_lords_goin_down_on_me.mp3','sounds/my-man.mp3', 'sounds/idubbbz-im-gay-free-download.mp3']

    playsound(sounds[np.random.randint(0,6)])



if __name__=='__main__':
    decision_tree('archive/data_trim_processed_1E5.csv')
