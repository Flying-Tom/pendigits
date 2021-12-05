from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


def DT(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # print("test loss:", metrics.accuracy_score(y_test, clf.predict(X_test)))
    # print("train loss:", metrics.accuracy_score(y_train, clf.predict(X_train)))

    y_pred = clf.predict(X_test)

    #print(f"Accuracy: {(100*metrics.accuracy_score(y_test, y_pred)):>0.1f}%")
    return metrics.accuracy_score(y_test, y_pred)
