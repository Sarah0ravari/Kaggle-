

                               


from imblearn.over_sampling import SMOTE
data = pd.read_csv("data.csv")

data.head(5)

# Dropping the ID column from data file
data.drop(columns=["ID"], inplace=True)
# remove all the null values
i = 0
for col in data.columns:
    if(data[col].isnull().any()):
        i += 1
print("No. of features that have null values:", i)

# Duplicated value that holds the same value
i = 0
for col in data.columns:
    if len(data[col].unique()) == 1:
        data.drop(columns=[col], inplace=True)
        i += 1
print(" No. of features that have constant values and removed:", i)

# we find the basic information of the data set here
data.describe()
# Let's check the unique values in var3 feature
print("Unique values in var3")
print(np.sort(data['var3'].unique()))

print("Value count for var3:")
data['var3'].value_counts().to_frame().T


data_cols = data.columns
# Reference: https://www.kaggle.com/cast42/debugging-var3-999999
# Taking all the datapoints where var3!=-999999 (without var3 feature) as the Xtrain
X_train = data.loc[data['var3'] != -999999, data.columns[1:]]
# Taking all the var3 values where var3!=-999999 as the Ytrain
y_train = data.loc[data['var3'] != -999999, 'var3']
X_test = data.loc[data['var3'] == -999999, data.columns[1:]]
# Taking all the datapoints where var3!=-999999 (without var3 feature) as the Xtest
# Defining KNN for predicitng the values to replace for -999999
clf = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)

# Replacing -999999 with 2
data['var3'].replace(to_replace=-999999, value=2, inplace=True)


dup = data.duplicated(keep='first')
# keep='first' to retain one instance of all the duplicate points
dup_count = data["TARGET"][dup].value_counts()
print("There are {} duplicate entries for TARGET=0(happy customers) and {} duplicate entries for TARGET=1(unhappy customers)".format(
    dup_count[0], dup_count[1]))
print("Total duplicate rows={}".format(dup_count[0]+dup_count[1]))


# Removing duplicate rows by keeping only one instance of all
dup_ind = data.index[dup]
data.drop(index=dup_ind, inplace=True)

# Checking if any rows have same features but different Target labels which is invalid
dup_list = data[data_cols[:-1]].duplicated(keep=False)
dup_count = data["TARGET"][dup_list].value_counts()
print("There are {} data points that have same feature values, but with different target labels".format(
    dup_count[0]+dup_count[1]))

# getting boolean representation of all duplicates
dup_total = data[data_cols[:-1]].duplicated(keep=False)
# getting boolean representation of first set of 133 duplicates
dup_first = data[data_cols[:-1]].duplicated(keep='first')
# getting boolean representation of second set of 133 duplicates
dup_last = data[data_cols[:-1]].duplicated(keep='last')
dup_total_index = data.index[dup_total]  # indeces of all duplicates
# creating train data without any duplicates
temp_x_train = data.drop(index=dup_total_index)
temp_y_train = temp_x_train["TARGET"]
# dropping target variable from train data
temp_x_train.drop(columns=["TARGET"], inplace=True)
dup_first_index = data.index[dup_first]  # indices of one set of 133 duplicates
# test data with one set of 133 duplicates
temp_xtest = data.loc[dup_first_index]
# dropping target variable from test data
temp_xtest.drop(columns=["TARGET"], inplace=True)
# Defining KNN for predicitng the values
clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
clf.fit(temp_x_train, temp_y_train)
pred = clf.predict(temp_xtest)

# Replacing the target values of one set of 133 duplicate rows with predicted value
for ind in dup_first_index:
    data.loc[ind, 'TARGET'] = 0
# Dropping the other set of 133 duplicates
dup_last_index = data.index[dup_last]  # indices of one set of 133 duplicates
data.drop(index=dup_last_index, inplace=True)

count = data["TARGET"].value_counts()
happy_percent = np.round(100*count[0]/(count[0]+count[1]), 2)
unhappy_percent = np.round(100*count[1]/(count[0]+count[1]), 2)
print("Number of happy customers={} and their percentage={}%".format(
    count[0], happy_percent))
print("Number of unhappy customers={} and their percentage={}%".format(
    count[1], unhappy_percent))

plt.bar([0, 1], [happy_percent, unhappy_percent])
plt.xticks([0, 1], ['Happy', 'Unhappy'])
plt.xlabel("Customer Category")
plt.ylabel("Percentage")
plt.title("Target distribution")
plt.show()

# spliting data to train and test set
X = data.drop(columns=["TARGET"])
Y = data["TARGET"]
X_train1, X_test, y_train1, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=0, stratify=Y)  # Splitting data into test and train datasets
count = y_train1.value_counts()
print("No. of data points before upsampling=", y_train1.shape[0])
print("Percentage of unhappy customers in train data before upsampling={}%".format(
    round(count[1]*100/(count[0]+count[1]), 2)))
print("Percentage of happy customers in train data before upsampling={}%".format(
    round(count[0]*100/(count[0]+count[1]), 2)))

# balancing data
smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train1, y_train1)  # upsampling using SMOTE
# SMOTE return numpy array. Converting it to dataframe
X_train = pd.DataFrame(X_train, columns=list(data_cols)[:-1])
y_train = pd.DataFrame(y_train, columns=["TARGET"])
count = y_train["TARGET"].value_counts()
print("No. of data points after upsampling=", y_train.shape[0])
print("Percentage of unhappy customers in train data before upsampling={}%".format(
    round(count[1]*100/(count[0]+count[1]), 2)))
print("Percentage of happy customers in train data before upsampling={}%".format(
    round(count[0]*100/(count[0]+count[1]), 2)))


#

happy_percent = np.round(100*count[0]/(count[0]+count[1]), 2)
unhappy_percent = np.round(100*count[1]/(count[0]+count[1]), 2)
plt.bar([0, 1], [happy_percent, unhappy_percent])
plt.xticks([0, 1], ['Happy', 'Unhappy'])
plt.xlabel("Customer Category")
plt.ylabel("Percentage")
plt.title("Target distribution after upsampling")
plt.show()


# Data Standardization
# Train data standardization
std = StandardScaler()
X_train = std.fit_transform(X_train)
# Test data standardization
X_test = std.transform(X_test)

# Dimention Reduction using PCA
print("Number of features in data=", X_train.shape[1])

# Since the number is very high we reduce diamenton with the PCA
pca = decomposition.PCA()
pca.n_components = X_train.shape[1]
pca_data = pca.fit_transform(X_train)
percentage_var_explained = pca.explained_variance_ / \
    np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)
# Plot the PCA spectrum
plt.figure(1, figsize=(8, 6))
plt.plot(cum_var_explained, linewidth=2)
plt.title("Explained_variance VS n_components")
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

# Plotting variance of 150 features
plt.figure(1, figsize=(8, 6))
plt.plot(cum_var_explained[0:150], linewidth=2)
plt.title("Explained_variance VS n_components")
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

# Therefore we reduce to 150
pca.n_components = 150
X_train = pca.fit_transform(X_train)  # PCA on train data
X_test = pca.transform(X_test)  # PCA on test data
print("Size of train data after dimensionality reduction(PCA)=", X_train.shape)
print("Size of test data after dimensionality reduction(PCA)=", X_test.shape)

# Describing the machine learning Model
# using Auc score to describe the best model by searhing
tuned_param = {'n_estimators': range(
    10, 20), 'learning_rate': [.05, .06, .07, .08, .09, .1, 0.15, 0.2, 0.25, 0.3]}
xgb_model = GridSearchCV(XGBClassifier(n_jobs=-1, random_state=0),
                         param_grid=tuned_param, scoring='roc_auc', return_train_score=True)
xgb_model.fit(X_train, y_train)
print("Best hyperparamters-", xgb_model.best_params_)
print("Best AUC value: ", xgb_model.best_score_)

# The idea of how model performence with different hyper premeter
auc_df = pd.DataFrame(xgb_model.cv_results_['params'])
# Creating a data frame with hyperparameters and AUC
auc_df["auc"] = xgb_model.cv_results_['mean_test_score']
# Pivoting the dataframe for plotting heat map
auc_df = auc_df.pivot(index='learning_rate',
                      columns='n_estimators', values='auc')
plt.figure(figsize=(20, 10))
sns.heatmap(data=auc_df, annot=True)
plt.title("AUC plot for CV data")
plt.show()

# You can visit the model performance here
pred_train_prob = xgb_model.predict_proba(X_train)
pred_test_prob = xgb_model.predict_proba(X_test)
print("Train AUC=", roc_auc_score(y_train, pred_train_prob[:, 1]))
print("Test AUC=", roc_auc_score(y_test, pred_test_prob[:, 1]))
pred_train = xgb_model.predict(X_train)
pred_test = xgb_model.predict(X_test)
# Plotting confusion, precision and recall matrices of train data
plot_confusion_matrix(y_train, pred_train)

# using logistic regression to have more reliable model
alpha_list = [10**a for a in range(-5, 3)]
train_auc_list = []
test_auc_list = []
for i in alpha_list:
    log_clf = LogisticRegression(penalty='l2', C=i, max_iter=500, n_jobs=-1)
    calib_clf = CalibratedClassifierCV(log_clf, method="sigmoid")
    calib_clf.fit(X_train, y_train)
    pred_train = calib_clf.predict_proba(X_train)
    pred_test = calib_clf.predict_proba(X_test)
    train_auc = roc_auc_score(y_train, pred_train[:, 1])
    test_auc = roc_auc_score(y_test, pred_test[:, 1])
    train_auc_list.append(train_auc)
    test_auc_list.append(test_auc)

    # we are going to plot AUC for hypre prameter
    for i in range(0, len(alpha_list)):
    print("For alpha=", alpha_list[i], " train AUC=",
          train_auc_list[i], " and test AUC=", test_auc_list[i])
plt.title("alpha VS AUC")
plt.plot(alpha_list, train_auc_list, label="Train")
plt.plot(alpha_list, test_auc_list, label="Test")
plt.xlabel("alpha")
plt.ylabel("AUC")
plt.legend()
plt.show()

# The best AUC value here
best_alpha_ind = np.argmax(test_auc_list)
print("Best alpha=", alpha_list[best_alpha_ind])
print("Best train AUC=", train_auc_list[best_alpha_ind])
print("Best test AUC=", test_auc_list[best_alpha_ind])
# XgBoost model to that for RFDT with just one line of code change.
tuned_param = {'max_depth': range(1, 10), 'n_estimators': range(1, 6)}
rfdt_model = GridSearchCV(RandomForestClassifier(
    n_jobs=-1), param_grid=tuned_param, scoring='roc_auc', return_train_score=True)
rfdt_model.fit(X_train, y_train)

# Print Auc value
print("Best hyperparamters-", rfdt_model.best_params_)
print("Best AUC value: ", rfdt_model.best_score_)
