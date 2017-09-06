#!/usr/bin/python

import operator
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances',
'bonus', 'restricted_stock_deferred', 'deferred_income','total_stock_value', 'expenses',
'exercised_stock_options',  'long_term_incentive', 'restricted_stock', 'director_fees',
'to_messages','from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Total number of data points:" + str(len(data_dict))
# print "Total number of Poi:" + str(data_dict['poi'].count(True))
num_poi = 0
num_nonpoi = 0
for keys in data_dict:
    if data_dict[keys]['poi']:
        num_poi += 1
    else:
        num_nonpoi += 1

print "allocation across classes (POI/non-POI):" + str(num_poi) + "/" + str(num_nonpoi)

### Find the number of NaN in each feature

for feature in features_list:
    num_Na=0
    num_data=0
    for keys in data_dict:
        if data_dict[keys][feature] == "NaN":
            num_Na +=1
            num_data += 1
        else:
            num_data +=1

    print feature
    print "number of NaN value:" + str(num_Na)
    print "number of data:" + str(num_data)
    print


features_list = ['poi','salary', 'total_payments', 'bonus',  'deferred_income','total_stock_value', 'expenses', 'exercised_stock_options',
'long_term_incentive', 'restricted_stock', 'director_fees','from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']


print "number of features used:" + str(len(features_list))
## Task 2: Remove outliers

### Look into data to see if there is any extreme outlier.
num_NAN = {}
for keys in data_dict:
    num_NAN[keys] = 0

for feature in features_list:
    outliers = {}
    for keys in data_dict:
        if data_dict[keys][feature] == "NaN":
            outliers[keys] = 0
            num_NAN[keys] += 1
        else:
            outliers[keys] = data_dict[keys][feature]
    outliers = sorted(outliers.iteritems(), key=operator.itemgetter(1), reverse=True)
    n = 0
    print
    print feature
    print

    for key, value in outliers:
        if n > 10:
            break
        else:
            print key, value
            n+=1


num_NAN = sorted(num_NAN.iteritems(), key=operator.itemgetter(1), reverse=True)
print
print "Person with most missing data"
print
n=0
for key, value in num_NAN:
    if n > 10:
        break
    else:
        print key, value
        n+=1


for keys, values in data_dict.items():
    if keys == "TOTAL":
        del data_dict[keys]
    elif keys == "THE TRAVEL AGENCY IN THE PARK":
        del data_dict[keys]
    elif keys == "LOCKHART EUGENE E":
        del data_dict[keys]


### Task 3: Create new feature(s)
for keys in data_dict:
    if data_dict[keys]["salary"] == "NaN" or data_dict[keys]["bonus"] == "NaN":
        data_dict[keys]["income"] = "NaN"
    else:
        data_dict[keys]["income"] = int(data_dict[keys]["salary"]) + int(data_dict[keys]["bonus"])

for keys in data_dict:
    if data_dict[keys]["from_poi_to_this_person"] == "NaN" or data_dict[keys]["from_this_person_to_poi"] == "NaN":
        data_dict[keys]["contact_poi"] = "NaN"
    else:
        data_dict[keys]["contact_poi"] = int(data_dict[keys]["from_poi_to_this_person"]) + int(data_dict[keys]["from_this_person_to_poi"])

features_list.append("income")
features_list.append("contact_poi")
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Use SelectKBest to choose which feature to use for machine learning

k=5
selKBest = SelectKBest(f_regression, k)

selKBest.fit(features, labels)

selKBest.transform(features).shape

mask = selKBest.get_support()

scores = selKBest.scores_
feature_score = zip(features_list[1:], scores)
feature_score = list(reversed(sorted(feature_score, key=lambda x: x[1])))

features=[]
scores = []
for feature, score in feature_score:
    features.append(feature)
    scores.append(float(score))
y_pos = np.arange(len(features))
plt.bar(y_pos, scores)
plt.xticks(y_pos, features)
plt.tick_params(axis='x', labelsize=5)
plt.show()
k_best_features = dict(feature_score[:k])
print
print "{0} best features: {1}\n".format(k, k_best_features.keys())
print k_best_features

data = featureFormat(my_dataset, k_best_features.keys(), sort_keys = True)
labels, features = targetFeatureSplit(data)

### scaling the feature

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features=scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf5 = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# clf5.fit(features_train, labels_train)
# print "Gaussian Naive Bayes test score:"
# print clf5.score(features_test, labels_test)
#
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(min_samples_split=5, criterion= 'gini', max_depth= 7, random_state = 42)
# clf.fit(features_train, labels_train)
# print "Decision Tree test score:"
# print clf.score(features_test, labels_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=2, criterion= 'gini', max_depth= 20, random_state = 25)
clf.fit(features_train, labels_train)
print "Decision Tree test score:"
print clf.score(features_test, labels_test)

# from sklearn import svm
# clf3 = svm.SVC()
# clf3.fit(features_train, labels_train)
# print "Support Vector Machine test score:"
# print clf3.score(features_test, labels_test)
#
# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(n_estimators=50, min_samples_split=5, random_state = 25)
# algo = clf.fit(features_train, labels_train)
# print "RandomForestClassifier test score:"
# print algo.score(features_test, labels_test)
# ### will do grid search to do cross validate
# from sklearn import grid_search
# #
# param_grid = {'n_estimators': [1, 25, 50, 100,200], 'min_samples_split':[2, 5,10,20,30], 'random_state': [42]}
# RFC = grid_search.GridSearchCV(RandomForestClassifier(), param_grid)
# RFC.fit(features_train, labels_train)
# print "Using grid search RandomForestClassifier test score:"
# print RFC.best_params_
# print RFC.score(features_test, labels_test)
#
# param_grid = {'max_depth':list(range(1,100)), 'criterion':["gini"], 'min_samples_split':list(range(2,50)), 'random_state': list(range(0,30))}
# tree = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
# tree.fit(features_train, labels_train)
# print "Using grid search Decision Tree test score:"
# print tree.best_params_
# print tree.score(features_test, labels_test)
# param_grid = {'max_depth':[1,10,20,40,50,100,200], 'criterion':["gini"], 'min_samples_split':[2,10,20,40,50,100,200], 'random_state': list(range(0,30))}
# tree = grid_search.GridSearchCV(tree.DecisionTreeClassifier(), param_grid)
# tree.fit(features_train, labels_train)
# print "Using grid search Decision Tree test score:"
# print tree.best_params_
# print tree.score(features_test, labels_test)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
