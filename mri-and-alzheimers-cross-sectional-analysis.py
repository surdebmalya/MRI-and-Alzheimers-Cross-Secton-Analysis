import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

home_cross_sectional = pd.read_csv('/kaggle/input/mri-and-alzheimers/oasis_cross-sectional.csv')

# 'M/F' : Gender
# 'Hand' : Hand that the person uses frequently -> 'All R'
# 'Age' : Age of the person (18-96)
# 'Educ' : Education Level [1 (poor) to 5 (excellent)]
# 'SES' : Socioeconomic Status [1 (highest status) to 5 (lowest status)]
# 'MMSE' : Mini Mental State Examination [(range is from 0 = worst to 30 = best)]
# 'CDR' : Clinical Dementia Rating [(0 = no dementia, 0.5 = very mild AD, 1 = mild AD, 2 = moderate AD)]
# 'eTIV' : Estimated Total Intracranial Volume
# 'nWBV' : Normalize Whole Brain Volume
# 'ASF' : Atlas Scaling Factor
# 'Delay' : to the number of days between two medical visits.

#--------------------------------------------------------------------------------------------------------------#

# 1. SES

# People gives rating from 1 to 5
# And there are lots of Null values

# 2. MMSE

# MMSE values lies between = 14-30 
# Consists of Null values
# When a person has trouble remembering, learning new things is termed as Cognitive impairment, and MMSE is a test to measure it.

# 3. CDR

# It measures the symptomes of Dementia 
# Dementia is not a single disease in itself, but a general term to describe symptoms of 
# impairment in memory, communication, and thinking.

# Composite Rating	Symptoms
# 0	none
# 0.5	very mild
# 1	mild
# 2	moderate
# 3	severe

# Here, the rating is between 0-2

# 4. eTIV

# 1123-1992
# It is the volumn of that person's internal skull

# 5. nWBV

# It is the volumn of whole brain
# 0.64-0.89 = measures in 0-1

# 6. ASF

# Indicating the relation of size between the image and atlas

figure = plt.figure(figsize=(20,55))

figure.add_subplot(12,3,1)
sns.countplot(x='Hand',data=home_cross_sectional)

figure.add_subplot(12,3,2)
sns.countplot(x='Delay',data=home_cross_sectional)

figure.add_subplot(12,3,3)
plt.scatter(x=home_cross_sectional['eTIV'], y=home_cross_sectional['ASF'], color=('yellowgreen'), alpha=0.5)
plt.xlabel('eTIV')
plt.ylabel('ASF')

non_related_features = ['ID', 'Hand', 'Delay', 'ASF']


home_cross_sectional.drop(non_related_features, axis=1, inplace =True)
sns.countplot(x='M/F',data=home_cross_sectional)
missing_val_count_by_column = (home_cross_sectional.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

figure = plt.figure(figsize=(20,55))

figure.add_subplot(12,3,1)
sns.countplot(x='SES',data=home_cross_sectional)
# 'SES' : Socioeconomic Status [1 (highest status) to 5 (lowest status)]

figure.add_subplot(12,3,2)
sns.countplot(x='Educ',data=home_cross_sectional)
# 'Educ' : Educational Level [1 (poor) to 5 (excellent)]

home_cross_sectional['SES'].replace(np.NaN, home_cross_sectional['SES'].mean(), inplace = True)
home_cross_sectional['Educ'].replace(np.NaN, home_cross_sectional['Educ'].median(), inplace = True)

figure = plt.figure(figsize=(20,55))

figure.add_subplot(12,3,1)
sns.countplot(x='SES',data=home_cross_sectional)
# 'SES' : Socioeconomic Status [1 (highest status) to 5 (lowest status)]

figure.add_subplot(12,3,2)
sns.countplot(x='Educ',data=home_cross_sectional)
# 'Educ' : Educational Level [1 (poor) to 5 (excellent)]

missing_val_count_by_column = (home_cross_sectional.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

figure = plt.figure(figsize=(20,55))
# MMSE = (range is from 0 = worst to 30 = best)

figure.add_subplot(12,3,1)
plt.scatter(x=home_cross_sectional['MMSE'], y=home_cross_sectional['M/F'], alpha=0.5)
plt.xlabel('MMSE')
plt.ylabel('M/F')
# Gender

figure.add_subplot(12,3,2)
plt.scatter(x=home_cross_sectional['MMSE'], y=home_cross_sectional['Age'], alpha=0.5)
plt.xlabel('MMSE')
plt.ylabel('Age')

figure.add_subplot(12,3,3)
plt.scatter(x=home_cross_sectional['MMSE'], y=home_cross_sectional['eTIV'], alpha=0.5)
plt.xlabel('MMSE')
plt.ylabel('eTIV')
# 'eTIV' : Estimated Total Intracranial Volume

figure.add_subplot(12,3,4)
plt.scatter(x=home_cross_sectional['MMSE'], y=home_cross_sectional['nWBV'], alpha=0.5)
plt.xlabel('MMSE')
plt.ylabel('nWBV')
# 'nWBV' : Normalize Whole Brain Volume

# using dummy values

home_cross_sectional = pd.get_dummies(home_cross_sectional)
home_cross_sectional.drop(['M/F_F'], axis=1, inplace=True)

features_for_MMSE = ['M/F_M', 'Age', 'eTIV', 'nWBV', 'MMSE']
data_for_MMSE_analysis = home_cross_sectional[features_for_MMSE]

train_data_for_MMSE = data_for_MMSE_analysis[data_for_MMSE_analysis['MMSE'].notnull()]
null_columns_for_MMSE = data_for_MMSE_analysis[data_for_MMSE_analysis['MMSE'].isnull()]
null_columns_for_MMSE.drop(['MMSE'], axis=1, inplace=True)

y = train_data_for_MMSE['MMSE']
X = train_data_for_MMSE.drop(['MMSE'], axis=1)

missing_val_count_by_column = (train_data_for_MMSE.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

missing_val_count_by_column = (null_columns_for_MMSE.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

n_neighbors = 1

model_for_MMSE = KNeighborsClassifier(n_neighbors = n_neighbors)
model_for_MMSE.fit(X, y)

final_result = model_for_MMSE.predict(null_columns_for_MMSE)


MMSE_columns_with_nulls = home_cross_sectional['MMSE'].isnull()
count = 0
for index, value in enumerate(MMSE_columns_with_nulls):
    if value:
        home_cross_sectional.at[index, 'MMSE'] = final_result[count]
        count+=1

train_data = home_cross_sectional[home_cross_sectional['CDR'].notnull()]

test_data = home_cross_sectional[home_cross_sectional['CDR'].isnull()]
test_data.drop(['CDR'], axis=1, inplace=True)

y = train_data['CDR']
X = train_data.drop(['CDR'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.1, random_state = 0)

models = {}
sns.countplot(x='CDR',data=home_cross_sectional)
def related_pred(predictions):
    base_score = [0, 0.5, 1, 2]
    final_pred_list = []
    for each_value in predictions:
        abs_values = []
        for i in range(len(base_score)):
            abs_values.append(abs(base_score[i]-each_value))
        minimum = min(abs_values)
        for _ in range(len(abs_values)):
            if abs_values[_] == minimum:
                index = _
        final_pred_list.append(base_score[index])
    return np.array(final_pred_list)

# Random Forest

model_1 = RandomForestRegressor(n_estimators=400, random_state=0)

model_1.fit(train_X, train_y)
predictions = model_1.predict(val_X)

predictions = related_pred(predictions)

random_forest_error = mean_absolute_error(val_y, predictions)
models['Random Forest'] = random_forest_error

# XGBoost

model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05)

model_2.fit(train_X, train_y)
predictions = model_2.predict(val_X)
predictions = related_pred(predictions)

xgboost_error = mean_absolute_error(val_y, predictions)
models['XGBoost'] = xgboost_error

# SVC

lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(train_y)

model_3 = SVC()

model_3.fit(train_X, training_scores_encoded)
predictions = model_3.predict(val_X)
predictions = related_pred(predictions)

svc_error = mean_absolute_error(val_y, predictions)
models['SVC'] = svc_error

# Logistic Regression

model_4 = LogisticRegression()
model_4.fit(train_X, training_scores_encoded)

predictions = model_4.predict(val_X)
predictions = related_pred(predictions)

logistic_error = mean_absolute_error(val_y, predictions)
models['Logistic Reg.'] = logistic_error

df = pd.DataFrame(list(models.items()),columns = ['Model Name','MAE']) 

final_predictions = model_2.predict(test_data)
final_predictions = related_pred(final_predictions)

final_data['Predicted_CDR'] = final_predictions

print(final_data)
