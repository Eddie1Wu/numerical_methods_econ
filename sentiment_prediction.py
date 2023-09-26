
###################################### ReadMe ############################################

# Simply run the entire script. The predicted MSEs are printed to the console.
# The graphs are saved to the same directory as this script.

##########################################################################################


import numpy as np
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt




###################################### Functions #########################################
def make_dummies(data, column):
	"""
	Creates dummy variable using a column of categorical data
	Args:
	data - the Pandas dataframe
	column - a string indicating column name
	"""
    
	out = pd.get_dummies(data[column], drop_first=True)
	data = data.drop(columns=column)
    
	return pd.concat([data, out], axis=1)



def cv_and_test(model, params, scoring, n_folds, fig_title, plotting, **kwargs):
	"""
	Runs n-fold cross validation, refits the model with the optimal hyperparameter, 
	and returns the predicted score on the held-out test set.
	Args:
	model - name of the model
	params - a dictionary of parameters to run GridSearchCV
	scoring - a string indicating the type of scoring
	n-folds - the number of folds for cross validation
	fig_title - title of the figure used for saving
	plotting - True if requires plotting, False otherwise
	**kwargs - the keyword arguments for the model
	"""

	reg = model(**kwargs)
	hyperparameters = [params]

	clf = GridSearchCV(reg, hyperparameters, scoring=scoring, cv=n_folds)
	clf.fit(X_train, y_train)

	if plotting == True:
		fig, ax = plt.subplots()
		ax.plot(np.log10(params["alpha"]), -clf.cv_results_["mean_test_score"])
		ax.set_xlabel("log(Alpha)")
		ax.set_ylabel("Mean square error")
		plt.savefig(fig_title)

	return -clf.score(X_train, y_train), -clf.score(X_test, y_test)




###################################### Load dataset ######################################

# Load data
df = pd.read_csv("../kiva_loans_sample.csv")
print(df.isna().sum()) # Check for columns with missing entries




###################################### Data preprocessing ################################

# Fill the missing entries of the translated descriptions column
df.loc[df["description_translated"].isnull(), "description_translated"] = df["description"]


# Drop the entries with negative values and the extreme outliers for "days_until_funded"
df = df[(df["days_until_funded"]>0) & (df["days_until_funded"]<300)]


# Obtain the sentiments of the translated descriptions using NLTK vader
analyzer = SentimentIntensityAnalyzer()
out = df["description_translated"].apply(analyzer.polarity_scores)
out = out.apply(pd.Series)
df = pd.concat([df, out], axis=1)


# Get the str len of loan_use as a variable. The hypothesis is longer string len indicates more detailed description.
df["loan_use_len"] = df["loan_use"].str.len()


# Generate dummy variables
var_list = ["original_language", "sector", "gender"]
for var in var_list:
    df = make_dummies(df, var)


# Generate country dummy
med_gdp_countries = ['United States', 'Costa Rica', 'Turkey', 'Israel', 'Chile', 'China', 'Panama']
df["dev_country"] = df["country"].isin(med_gdp_countries)  # This is a binary variable, 1 indicates country with per capita GDP more than $10,000


# Standarize the non-categorical variables
col_names = ["loan_amount", "repayment_term", "neg", "neu", "pos", "loan_use_len"]
out = df[col_names]
out = StandardScaler().fit_transform(out.values)
df[col_names] = out


# Demean the dependent variable
df["days_until_funded"] = df["days_until_funded"] - df["days_until_funded"].mean()


# Define the feature matrix X
regressors = list(df.columns)
regressors = [e for e in regressors if e not in ('id','name','description','description_translated',\
													'activity','loan_use','country','town','currency',\
													'posted_time','days_until_funded','pictured','compound')]
X = df[regressors]


# Define the target vector y
y = df["days_until_funded"]


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
								    X,
								    y,
								    test_size=1/5,
								    random_state=5)
print("The shape of train and test datasets are:")
print(X_train.shape)
print(X_test.shape)




###################################### Tasks #############################################

### Run Lasso
alphas = np.logspace(-3.1,-1,100)
train_err, test_err = cv_and_test(Lasso, {"alpha": alphas}, "neg_mean_squared_error", 5, "lasso_cv.png", True, random_state=1, max_iter=50000)
print(f"The train mean squared error for Lasso is {train_err}.")
print(f"The test mean squared error for Lasso is {test_err}.")


### Run Ridge Regression
alphas = np.logspace(-3,2,100)
train_err, test_err = cv_and_test(Ridge, {"alpha": alphas}, "neg_mean_squared_error", 5, "ridge_cv.png", True)
print(f"The train mean squared error for Ridge is {train_err}.")
print(f"The test mean squared error for Ridge is {test_err}.")


### Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
train_err = mean_squared_error(y_train, rf.predict(X_train))
test_err = mean_squared_error(y_test, rf.predict(X_test))
print(f"The train mean squared error for RandomForestRegressor is {train_err}.")
print(f"The test mean squared error for RandomForestRegressor is {test_err}.")




# ### Use this for tuning random forest if needed
# # Max depth of the tree
# max_depth = [10, 20, 30]
# max_depth.append(None)
# # Min number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Min number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# grid = {"max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf}
# out = cv_and_test(RandomForestRegressor, grid, "neg_mean_squared_error", 5, "rf_cv.png", False)
# print(f"The predicted mean squared error for RandomForestRegressor is {-out}.")




# ### This part demonstrates how the cv_and_test() function works. Simply uncomment and run.
# lasso = Lasso(random_state=1, max_iter=10000)
# alphas = np.logspace(-3, -1, 100)
# hyperparameters = [{"alpha": alphas}]
# n_folds = 5
# score = "neg_mean_squared_error"

# clf = GridSearchCV(lasso, hyperparameters, scoring=score, cv=n_folds)
# clf.fit(X_train, y_train)

# plt.plot(alphas, -clf.cv_results_["mean_test_score"])
# plt.savefig("lasso_cv.png")

# clf.score(X_test, y_test)












