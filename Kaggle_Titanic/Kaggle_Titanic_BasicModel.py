import pandas as pd

# Data Extraction
path_traindata = "/deeplearning/KaggleProjects/titanic/train.csv"
path_testdata = "/deeplearning/KaggleProjects/titanic/test.csv"
traindata = pd.read_csv(path_traindata)
testdata = pd.read_csv(path_testdata)

# print(traindata.info())
# print(traindata.describe())
# print(testdata.info())
# print(testdata.describe())

data_all = pd.concat([traindata,testdata], ignore_index = True)
# print(data_all.info())
# print(data_all.describe())

# Data Cleaning
## Imputing Missing Values
data_all['Age'].fillna(data_all['Age'].mean(), inplace = True)
# print(data_all['Age'].isnull().any())

data_all['Fare'].fillna(data_all['Fare'].mean(), inplace = True)
# print(data_all['Fare'].isnull().any())

# print(data_all[data_all[['Embarked']].isnull().any(axis=1)])
# print(data_all['Embarked'].value_counts())
data_all['Embarked'].fillna('S',inplace=True)
# print(data_all['Embarked'].isnull().any())

data_all.drop('Cabin',axis=1,inplace=True)
# print(data_all.isnull().any())


## Feature Engineering
# print(data_all['Ticket'])
# print(data_all['Name'])
data_all.drop('Ticket',axis=1,inplace=True)
data_all.drop('Name',axis=1,inplace=True)

data_all['FamilySize'] = data_all['SibSp'] + data_all['Parch'] + 1
# print(data_all['FamilySize'])

data_all['Age'] = pd.cut(data_all['Age'], 5)
# cut divides the data into a specified number of equi-width intervals
# print(data_all['Age'].value_counts())

# print(data_all['Fare'].describe())

# print(data_all['Fare'].describe(percentiles = [0.6,0.9,0.98]))
# print(data_all[data_all['Fare'] > 300])

bins = [0, 14, 78, 220, 500, 600]
labels = ['VeryLow','Low', 'Middle', 'High', 'VeryHigh']
data_all['Fare'] = pd.cut(data_all['Fare'], bins=bins, labels=labels, right=False)
# print(data_all['Fare'].value_counts())

## One Hot Encoding
from sklearn.preprocessing import LabelEncoder

map_features = ['Sex', 'Pclass', 'Age', 'Fare','Embarked']
for feature in map_features:
    data_all[feature] = LabelEncoder().fit_transform(data_all[feature])
    
map_features_2 = ['Pclass','Sex','Embarked','FamilySize','Fare','Age']
encoded_features = pd.get_dummies(data_all[map_features_2], columns=map_features_2)
# print(encoded_features.info())

# 数据筛选
train_x = encoded_features.iloc[:traindata.shape[0]]
test_x = encoded_features.iloc[traindata.shape[0]:]
train_y = data_all['Survived'].iloc[:traindata.shape[0]]


# Model
'''
# Linear Regression
from sklearn.linear_model import LinearRegression
model_LinearRegression = LinearRegression()
model_LinearRegression.fit(train_x, train_y)
test_y = model_LinearRegression.predict(test_x)
threshold = 0.5
test_y = (test_y > threshold).astype(int)
print("Predicted y:", test_y)

predictions_df = pd.DataFrame({'PassengerId': testdata['PassengerId'], 'Survived': test_y})
output_file = "/deeplearning/KaggleProjects/titanic/Output/Predictions_LinearRegression.csv"
predictions_df.to_csv(output_file, index=False)
print("Predictions saved to:", output_file)
'''

'''
# Logistic Regression
from sklearn.linear_model import LogisticRegression
model_LogisticRegression = LogisticRegression()
model_LogisticRegression.fit(train_x, train_y)
test_y = model_LogisticRegression.predict(test_x).astype(int)
print("Predicted y:", test_y)
predictions_df = pd.DataFrame({'PassengerId': testdata['PassengerId'], 'Survived': test_y})
output_file = "/deeplearning/KaggleProjects/titanic/Output/Predictions_LogisticRegression.csv"
predictions_df.to_csv(output_file, index=False)
print("Predictions saved to:", output_file)
'''

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_RandomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=42)
model_RandomForestClassifier.fit(train_x, train_y)
test_y = model_RandomForestClassifier.predict(test_x).astype(int)
print("Predicted y:", test_y)
predictions_df = pd.DataFrame({'PassengerId': testdata['PassengerId'], 'Survived': test_y})
output_file = "/deeplearning/KaggleProjects/titanic/Output/Predictions_RandomForestClassifier.csv"
predictions_df.to_csv(output_file, index=False)
print("Predictions saved to:", output_file)
