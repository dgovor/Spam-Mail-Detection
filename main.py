#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset.
data_set = pd.read_csv('data/mail_data.csv')

# Check for any missing values. Drop if any.
data_set = data_set.dropna()

# Encode 'ham' as 0 and 'spam' as 1.
data_set['Category'].replace({'ham':0,'spam':1},inplace=True)

# Splitting dataset into input data and labels.
X, Y = data_set['Message'], data_set['Category']

# Splitting dataset into training (80%) and testing (20%) subsets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=34)