# %%

import matplotlib.pyplot as plt
import numpy as np
from algorithms.SVM import SVM
from algorithms.DT import DT
from data_utils import get_data
from imblearn.over_sampling import RandomOverSampler

re_sample = False

ros = RandomOverSampler(random_state=0)

# %% Noise Level Influence on Accuracy

plt.title('Noise Level Influence on Accuracy')
x = np.linspace(0, 1, 101)
ySVM, yDT = [], []

for i in x:
    X_train, y_train = get_data(train=True, corrupt=True, noise_level=i)
    if(re_sample == True):
        X_train, y_train = ros.fit_resample(X_train, y_train)
    X_test, y_test = get_data(train=False, corrupt=False)
    ySVM.append(SVM(X_train, y_train, X_test, y_test))
    yDT.append(DT(X_train, y_train, X_test, y_test))


plt.plot(x, ySVM, color='cyan', label='SVM')
plt.plot(x, yDT, color='blue', label='Decision Tree')
plt.legend()
plt.xlabel('Noise Level')
plt.ylabel('Accuracy')

plt.savefig('figs/noise_level.png')
# plt.show()
plt.close()

# %% Imbalance Ratio Influence on Accuracy

plt.title("Imbalance Ratio Influence on Accuracy")
x = np.linspace(1, 101, 101)
ySVM, yDT = [], []

for i in x:
    X_train, y_train = get_data(train=True, corrupt=True,  imb_ratio=(int)(i))
    if(re_sample == True):
        X_train, y_train = ros.fit_resample(X_train, y_train)
    X_test, y_test = get_data(train=False, corrupt=False)
    ySVM.append(SVM(X_train, y_train, X_test, y_test))
    yDT.append(DT(X_train, y_train, X_test, y_test))


plt.plot(x, ySVM, color='cyan', label='SVM')
plt.plot(x, yDT, color='blue', label='Decision Tree')
plt.legend()
plt.xlabel('Imbalance Ratio')
plt.ylabel('Accuracy')

plt.savefig('figs/imb_ratio.png')
# plt.show()
plt.close()
