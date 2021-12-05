from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def SVM(X_train, y_train, X_test, y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # clf = SVC(gamma="auto")
    # clf = SVR(kernel="poly",  gamma="auto", degree=3, epsilon=0.1, coef0=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    #print(f"Test Error|| Accuracy: {100*metrics.accuracy_score(y_test, y_pred):>0.1f}% \n")
    return metrics.accuracy_score(y_test, y_pred)
