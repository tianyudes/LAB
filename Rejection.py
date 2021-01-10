# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

testpath = 'D:/graduate/10/noise/clean/test.csv'
data_set = pd.read_csv('train.csv')
features = ['MAX-SEC', 'SEC-TRD','TRD-FRT']
X_train,X_valid,Y_train,Y_valid = train_test_split(data_set.loc[: ,features], data_set['Label'], test_size=0.3, stratify=data_set['Label'], random_state=66)
test = pd.read_csv(testpath)
print(test.head(10))

x_test = test.loc[:,features]
y_test = test['Label2']


def plot_feature_diabetes(model, features):
    plt.figure(figsize=(3,3))
    n_feature = 3
    plt.barh(range(n_feature), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_feature), features)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_feature)


from sklearn.neighbors import KNeighborsClassifier
n_neighbors = range(1, 11)
train_accuracy = []
valid_accuracy = []

#  n = 5的时候最好
for n_neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    knn.fit(X_train,Y_train)
    train_accuracy.append(knn.score(X_train,Y_train))
    valid_accuracy.append(knn.score(X_valid,Y_valid))


plt.plot(n_neighbors,train_accuracy,label='Training accuracy')
plt.plot(n_neighbors,valid_accuracy,label='Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, Y_train)

y_pred = tree.predict(X_valid)
y_valid = Y_valid.values

print("-------------- Decision Tree -------------------")

print("accuracy_score:{:.4f}".format(accuracy_score(y_valid, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_valid, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_valid, y_pred)))

print("f1_score:{:.4f}".format( metrics.f1_score(y_valid, y_pred)))

y_pred = tree.predict(x_test)

print("accuracy_score:{:.4f}".format(accuracy_score(y_test, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_test, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_test, y_pred)))

print("f1_score:{:.4f}".format(metrics.f1_score(y_test, y_pred)))

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(X_train,Y_train)

y_pred = rf.predict(X_valid)
y_valid = Y_valid.values

print("-------------- Random Forest Classifier -------------------")


print("accuracy_score:{:.4f}".format(accuracy_score(y_valid, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_valid, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_valid, y_pred)))

print("f1_score:{:.4f}".format( metrics.f1_score(y_valid, y_pred)))

y_pred = rf.predict(x_test)

print("accuracy_score:{:.4f}".format(accuracy_score(y_test, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_test, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_test, y_pred)))

print("f1_score:{:.4f}".format(metrics.f1_score(y_test, y_pred)))

print('-----------RFC————————')



from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,Y_train)

y_pred = svc.predict(X_valid)
y_valid = Y_valid.values
print("accuracy_score:{:.4f}".format(accuracy_score(y_valid, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_valid, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_valid, y_pred)))

print("f1_score:{:.4f}".format( metrics.f1_score(y_valid, y_pred)))

y_pred = svc.predict(x_test)

print("accuracy_score:{:.4f}".format(accuracy_score(y_test, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_test, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_test, y_pred)))

print("f1_score:{:.4f}".format(metrics.f1_score(y_test, y_pred)))


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.fit_transform(X_valid)
X_test_scaled = scaler.fit_transform(x_test)
svc = SVC(kernel="linear",C=15)
svc.fit(X_train_scaled,Y_train)

y_pred = svc.predict(X_valid_scaled)
y_valid = Y_valid.values

print("accuracy_score:{:.4f}".format(accuracy_score(y_valid, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_valid, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_valid, y_pred)))

print("f1_score:{:.4f}".format( metrics.f1_score(y_valid, y_pred)))

y_pred = svc.predict(X_test_scaled)

print("accuracy_score:{:.4f}".format(accuracy_score(y_test, y_pred)))

print("precision_score:{:.4f}".format(metrics.precision_score(y_test, y_pred)))

print("recall_score:{:.4f}".format(metrics.recall_score(y_test, y_pred)))

print("f1_score:{:.4f}".format(metrics.f1_score(y_test, y_pred)))

sum1 = y_pred.sum()
print(sum1)
y_pred = y_pred.tolist()

test["rejection_pre"] = y_pred
test["recognition"] = test.idxmax(axis=1)
savepath = testpath.replace("test.csv","restest.csv")
test.to_csv(savepath,index=False)



