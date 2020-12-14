import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from playsound import playsound

from sklearn.metrics import plot_confusion_matrix, accuracy_score

def XGBoost(filename, eta):

    data = pd.read_csv('archive/' +filename,
        usecols=['label', 'tweet']
    )
    print(data)

    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(data['tweet'])
    vectorized=vectorized.todense()
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    xgb_model = xgb.XGBClassifier(eta = eta, nthread = 4).fit(X_tr, y_tr)
    y_pred = xgb_model.predict(X_te)
    y_pred_tr = xgb_model.predict(X_tr)

    accuracy = accuracy_score(y_te, y_pred)
    accuracy_train = accuracy_score(y_tr, y_pred_tr)
    print(accuracy)
    print(accuracy_train)

    sounds = ['sounds/Not_Gay_Sex.mp3', 'sounds/Objection_Heresay.mp3','sounds/Rock_Flag_and_Eagle.mp3', 'sounds/The_good_lords_goin_down_on_me.mp3','sounds/my-man.mp3', 'sounds/idubbbz-im-gay-free-download.mp3']

    playsound(sounds[np.random.randint(0,6)])

    return accuracy, accuracy_train




if __name__ == '__main__':

    etas = [0.1, 0.3, 0.6, 0.8, 1, 1.3]
    accuracy = np.zeros(len(etas))
    accuracy_train = np.zeros(len(etas))
    for i, eta in enumerate(etas):

        accuracy[i], accuracy_train[i]=XGBoost('data_trim_edit_1E4.csv', eta)

    plt.style.use("ggplot")
    plt.plot(etas, accuracy_train, label='train')
    plt.plot(etas, accuracy, label='test')
    plt.title('XGBoost', size=16)
    plt.xlabel('Learning rate', size=14)
    plt.ylabel('Accuracy score', size=14)
    plt.legend()
    plt.show()
