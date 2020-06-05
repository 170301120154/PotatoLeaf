import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


datasets = pd.read_csv('Labled_DATAUpdate1.csv')

X = datasets.iloc[:,1:].values
print(X.shape)
Y = datasets.iloc[:, 0].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)


clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Score: ")
print(accuracy_score(y_test, y_pred)*100)

cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

# print('Report : ')
# print(classification_report(y_test, y_pred))
#
# sfs1 = sfs(clf,
#            k_features=5,
#            forward=True,
#            floating=False,
#            verbose=2,
#            scoring='accuracy',
#            cv=5)
#
# # Perform SFFS
# sfs1 = sfs1.fit(X_train, y_train)
