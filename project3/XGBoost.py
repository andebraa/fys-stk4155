import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


from sklearn.metrics import plot_confusion_matrix, accuracy_score

def XGBoost(filename, depth = 6):

    data = pd.read_csv('archive/' +filename,
        usecols=['label', 'tweet']
    )

    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(data['tweet'])
    vectorized=vectorized.todense()

    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    xgb_model = xgb.XGBClassifier(max_depth = depth, nthread = 4).fit(X_tr, y_tr)
    y_pred = xgb_model.predict(X_te)
    y_pred_tr = xgb_model.predict(X_tr)

    accuracy = accuracy_score(y_te, y_pred)
    accuracy_train = accuracy_score(y_tr, y_pred_tr)

    return accuracy, accuracy_train




if __name__ == '__main__':

    depths = [1, 3, 4, 6, 11, 15]
    accuracy = np.zeros(len(depths))
    accuracy_train = np.zeros(len(depths))
    for i, depth in enumerate(depths):

        accuracy[i], accuracy_train[i]=XGBoost('data_trim_edit_1E4.csv', depth)

    plt.style.use("ggplot")
    plt.plot(depths, accuracy_train, label='train')
    plt.plot(depths, accuracy, label='test')
    plt.title('XGBoost', size=16)
    plt.xlabel('Maximum tree depth', size=14)
    plt.ylabel('Accuracy score', size=14)
    plt.legend()
    plt.show()
