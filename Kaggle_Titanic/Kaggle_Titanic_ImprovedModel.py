import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# sns.countplot(x='Sex', data=traindata)
# plt.title('Distribution of Sex')
# plt.savefig('/deeplearning/KaggleProjects/titanic/Photoes/Sex_Numbers.png')
# plt.show()

# sns.barplot(x = 'Sex', y='Survived', data = traindata)
# plt.title('Sex_Survived')
# plt.savefig('/deeplearning/KaggleProjects/titanic/Photoes/Sex_Survived.png')
# plt.show()

# print(traindata['Sex'].value_counts().values/traindata.shape[0])

# plt.hist([data_all[data_all['Survived']==1]['Pclass'],data_all[data_all['Survived']==0]['Pclass']], color = ['g','r'], label = ['Survived','Dead'])
# plt.title('Pclass_Survived')
# plt.xlabel('Pclass')
# plt.ylabel('Numbers')
# plt.legend()
# plt.savefig('/deeplearning/KaggleProjects/titanic/Photoes/Pclass_Survived.png')
# plt.show()

# plt.hist([data_all[data_all['Survived']==1]['Fare'],data_all[data_all['Survived']==0]['Fare']],color = ['g','r'], bins = 50, label = ['Survived','Dead'])
# plt.title('Fare_Survived')
# plt.xlabel('Fare')
# plt.ylabel('Numbers')
# plt.legend()
# plt.savefig('/deeplearning/KaggleProjects/titanic/Photoes/Fare_Survived.png')
# plt.show()

# print(data_all['Cabin'].unique())

data_all['Cabin'].fillna('U',inplace=True)
data_all['Cabin'] = data_all['Cabin'].map(lambda s: s[0])
# print(data_all['Cabin'].value_counts())

# print(data_all[data_all['Cabin']=='T'])
data_all.loc[(data_all['Cabin'] == 'T'), 'Cabin'] = 'F'

# print(data_all.groupby("Cabin")['Fare'].max().sort_values())
# print(data_all.groupby("Cabin")['Fare'].min().sort_values())
# print(data_all.groupby("Cabin")['Fare'].mean().sort_values())

# print(data_all[data_all['Fare'] == 0])
data_all.loc[(data_all['Fare'] == 0) & (data_all['Pclass'] == 1) &(data_all['Cabin'] == 'U'), 'Cabin'] = 'B'
# print(data_all[data_all['Fare'] == 0])

# print(data_all.groupby("Cabin")['Fare'].mean().sort_values())

def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

data_all.loc[data_all['Cabin'] == 'U','Cabin'] = data_all[data_all['Cabin'] == 'U']['Fare'].apply(lambda x: cabin_estimator(x))
print(data_all['Cabin'].value_counts())

data_all['Cabin'] = data_all['Cabin'].replace(['A','B'], 'AB')
data_all['Cabin'] = data_all['Cabin'].replace(['C','D','E'], "CDE")
data_all['Cabin'] = data_all['Cabin'].replace(['F','G'], 'FG')


data_all['Title'] = data_all['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
# print(data_all['Title'].unique())
data_all['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mrs'],inplace=True)

# print(data_all.groupby("Title")['Age'].mean().sort_values())

data_all.loc[(data_all['Age'].isnull())&(data_all['Title']=='Mr'),'Age']=33
data_all.loc[(data_all['Age'].isnull())&(data_all['Title']=='Mrs'),'Age']=37
data_all.loc[(data_all['Age'].isnull())&(data_all['Title']=='Master'),'Age']=5
data_all.loc[(data_all['Age'].isnull())&(data_all['Title']=='Miss'),'Age']=22
data_all.loc[(data_all['Age'].isnull())&(data_all['Title']=='Other'),'Age']=45

# same process
data_all['Embarked'].fillna('S',inplace=True)
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

map_features = ['Cabin','Sex', 'Title', 'Age', 'Fare','Embarked']
for feature in map_features:
    data_all[feature] = LabelEncoder().fit_transform(data_all[feature])
# print(data_all.info())
map_features_2 = ['Pclass','Sex','Title','Cabin','Embarked','FamilySize','Fare','Age']
encoded_features = pd.get_dummies(data_all[map_features_2], columns=map_features_2)
# print(encoded_features.info())

# 数据筛选
train_x = encoded_features.iloc[:traindata.shape[0]]
test_x = encoded_features.iloc[traindata.shape[0]:]
train_y = data_all['Survived'].iloc[:traindata.shape[0]]

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model_LogisticRegression = LogisticRegression()
model_LogisticRegression.fit(train_x, train_y)
test_y = model_LogisticRegression.predict(test_x).astype(int)
print("Predicted y:", test_y)
predictions_df = pd.DataFrame({'PassengerId': testdata['PassengerId'], 'Survived': test_y})
output_file = "/deeplearning/KaggleProjects/titanic/Output/Predictions_LogisticRegression_Improved.csv"
predictions_df.to_csv(output_file, index=False)
print("Predictions saved to:", output_file)