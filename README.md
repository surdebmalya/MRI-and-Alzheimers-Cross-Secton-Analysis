# MRI-and-Alzheimers-Cross-Secton-Analysis
In this Machine Learning project, I have worked on a Kaggle dataset of 'MRI and MRI and Alzheimers'.
Here I have worked on 4 machine learning models:
- RandomForestRegressor
- XGBRegressor
- SVC
- LogisticRegression
Among them XGBRegressor is giving best result, so I used it to generate final dataframe as result, named 'final_data'.

# By using XGBRegressor on model, I got a MAE of 0.145833, which was less among those for algorithms I have used. So I used it on test_data to generate final_data.

# Remark
In this model I have to face the problem of data deficiency. There is lots of Null values(NaN) in the Dependent Feature called 'CDR'. Among 436 total data, 201 Dependent Feature are NaN, so I was forced to take them as test_data set, so my test_data set becomes 201 rows, and among the rest of data, I used 24 rows for validation. So, I have faced data deficiency problem. If there was more data, then I could analyze the skewness and also outliers could be analyzed if there was more data.
