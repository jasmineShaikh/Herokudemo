# Importing the libraries

import pandas as pd
import pickle

dataset = pd.read_csv('HR_comma_sep.csv')
dataset['satisfaction_level'] = dataset['satisfaction_level'].apply (lambda x: x*100)
dataset['last_evaluation'] = dataset['last_evaluation'].apply (lambda x: x*100)
print(dataset['sales'].unique())
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels in column 'species'.
dataset['sales'] = label_encoder.fit_transform(dataset['sales'])
dataset['sales'].unique()
dataset['salary'] = label_encoder.fit_transform(dataset['salary'])
dataset['salary'].unique()
print(dataset['sales'].head(10))
print(dataset['satisfaction_level'])

dataset=dataset.astype(int)
X = dataset.iloc[:, :9]

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.


from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report

X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.20)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
predicted_xg = rf.predict(X_test)
# report = classification_report(y_test, predicted)
# print(report)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predicted = knn.predict(X_test)
report = classification_report(y_test, predicted)
print(report)

# Saving model to disk
pickle.dump(knn, open('model.pkl','wb'))

from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train,y_train)
predicted = xg.predict(X_test)
report = classification_report(y_test, predicted_xg)
print(report)



from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)
predicted = gb.predict(X_test)
report = classification_report(y_test, predicted)
print(report)

pickle.dump(rf, open('model1.pkl','wb'))
pickle.dump(gb, open('model2.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))

print(model1.predict([[38,53,2,157,3,0,0,7,1]]))