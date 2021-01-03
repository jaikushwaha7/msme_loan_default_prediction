# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('MSME data 4 var for Deployment.csv')

#dataset['sme_category'].fillna('Missing', inplace=True)

#dataset['asset_cost'].fillna(dataset['asset_cost'].mean(), inplace=True)

#dataset['ltv'].fillna(dataset['ltv'].mean(), inplace=True)

df_cols = [col for col in dataset.columns.tolist() if dataset[col].dtype in ['object']] 

#float_cols = [col for col in dataset.columns.tolist() if dataset[col].dtype in ['object']] 
df_copy = dataset.copy()

# feature engineering
for cols in df_cols:
    prob_df=df_copy.groupby([cols])['loan_default'].mean()
    prob_df=pd.DataFrame(prob_df)
    prob_df['no_default']=1-prob_df['loan_default']
    prob_df['Probability_ratio']=prob_df['loan_default']/(prob_df['no_default']+.00001)
    probability_encoded=prob_df['Probability_ratio'].to_dict()
    df_copy[cols]=df_copy[cols].map(probability_encoded)

X = df_copy.iloc[:,:4]


y = df_copy.iloc[:, -1]



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

#Fitting model with trainig data
logreg.fit(X, y)

# Saving model to disk
pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))