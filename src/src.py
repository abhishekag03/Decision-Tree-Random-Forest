#!/usr/bin/env python
# coding: utf-8

# In[335]:


import pandas
import numpy
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


# In[439]:


def calculate_mean_std(array, axis):
    return numpy.mean(array, axis = axis), numpy.std(array, axis = 1)


# In[300]:


train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")
train_data = numpy.array(train_data)
test_data = numpy.array(test_data)
x_train = train_data[:,2:]
y_train = train_data[:,1]
x_test = test_data[:,2:]
y_test = test_data[:,1]


# In[14]:


print(x_train)


# In[147]:


decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(x_train, y_train)
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)


# In[148]:


pred = decision_tree_classifier.predict(x_test)
print(accuracy_score(pred, y_test))
print(decision_tree_classifier.score(x_train, y_train))
pred2 = random_forest_classifier.predict(x_test)
print(accuracy_score(pred2, y_test))
print(random_forest_classifier.score(x_train, y_train))


# In[489]:


parameter_grid = {'criterion': ["gini", "entropy"], 'max_depth' : list(numpy.arange(1, 10, 1)), 'min_samples_split' : numpy.arange(10, 30, 5), 'min_samples_leaf' : [1,3,7,10]}
decision_tree_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid = parameter_grid, cv = 3)
decision_tree_classifier.fit(x_train, y_train)
print(decision_tree_classifier.score(x_train, y_train))
print(decision_tree_classifier.score(x_test, y_test))


# In[446]:


print(decision_tree_classifier.best_params_)


# In[449]:


pickle_out = open("Decision_Tree_Classifier.pickle","wb")
pickle.dump(decision_tree_classifier, pickle_out)
pickle_out.close()


# In[334]:


parameter_grid_rf = {'criterion': ["gini", "entropy"], 'max_depth' : list(numpy.arange(3, 30, 5)), 'min_samples_split' : numpy.arange(50, 150, 20), 'min_samples_leaf' : numpy.arange(20, 100, 20), 'n_estimators' : numpy.arange(10, 100, 20)}
random_forest_classifier = GridSearchCV(RandomForestClassifier(), param_grid = parameter_grid_rf, cv = 3)
random_forest_classifier.fit(x_train, y_train)
pred_train = random_forest_classifier.predict(x_train)
pred = random_forest_classifier.predict(x_test)
print(accuracy_score(pred_train, y_train))
print(accuracy_score(pred, y_test))


# In[443]:


results = pandas.DataFrame(decision_tree_classifier.cv_results_)
results.to_csv("Decision_tree_results.csv")
results = pandas.DataFrame(random_forest_classifier.cv_results_)
results.to_csv("Random_forest_results.csv")


# In[336]:


print(random_forest_classifier.best_params_)


# In[451]:


pickle_out = open("Random_Forest_Classifier.pickle","wb")
pickle.dump(random_forest_classifier, pickle_out)
pickle_out.close()


# In[441]:


best_params_rf = random_forest_classifier.best_params_
rf_classifier = RandomForestClassifier(n_estimators=best_params_rf['n_estimators'], criterion=best_params_rf['criterion'], max_depth=best_params_rf['max_depth'], min_samples_split=best_params_rf['min_samples_split'], min_samples_leaf=best_params_rf['min_samples_leaf'])
rf_classifier.fit(x_train, y_train)
rf_classifier.score(x_train, y_train)


# In[425]:


plt.grid()
train_sizes, train_scores, test_scores = learning_curve(rf_classifier, x_train, y_train, cv = 3, train_sizes = numpy.linspace(.1, 1.0, 5))
print(train_scores)
train_scores_mean, train_score_std = calculate_mean_std(train_scores, 1)
test_scores_mean, test_scores_std = calculate_mean_std(test_scores, 1)
plt.plot(train_sizes, train_scores_mean, 'o-', color = "r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color = "g", la1bel="Test score")
plt.legend(loc="best")
plt.show()
#Source for plotting learning curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html


# In[355]:


best_params_dt = decision_tree_classifier.best_params_
dt_classifier = DecisionTreeClassifier(criterion=best_params_dt['criterion'], max_depth=best_params_dt['max_depth'], min_samples_split=best_params_dt['min_samples_split'], min_samples_leaf=best_params_dt['min_samples_leaf'])
dt_classifier.fit(x_train, y_train)
dt_classifier.score(x_train, y_train)


# In[491]:


results_dt = decision_tree_classifier.cv_results_
test_score_dt = results_dt['mean_test_score']
ind = numpy.argmax(test_score_dt)
print(ind)
errors_dt = []
for i in range(3):
    errors_dt.append(1-results['split'+str(i)+'_test_score'][ind])
print(errors_dt)
print(numpy.var(errors_dt))


# In[490]:


results_rf = random_forest_classifier.cv_results_
test_score = results_rf['mean_test_score']
ind = numpy.argmax(test_score)
print(ind)
errors = []
for i in range(3):
    errors.append(1-results['split'+str(i)+'_test_score'][ind])
print(errors)
print(numpy.var(errors))


# In[488]:


decision_tree_classifier.cv_results_


# In[452]:


pickle_out = open("classifier_dt.pickle","wb")
pickle.dump(dt_classifier, pickle_out)
pickle_out.close()
pickle_out = open("classifier_rf.pickle","wb")
pickle.dump(rf_classifier, pickle_out)
pickle_out.close()


# In[442]:


print(best_params_rf)


# In[437]:


plt.grid()
train_sizes, train_scores, test_scores = learning_curve(dt_classifier, x_train, y_train, cv = 3, train_sizes = numpy.linspace(.1, 1.0, 5))
print(train_scores)
train_scores_mean, _ = calculate_mean_std(train_scores, 1)
test_scores_mean, _ =  calculate_mean_std(test_scores, 1)
plt.plot(train_sizes, train_scores_mean, 'o-', color = "r",
             label = "Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label = "Test score")
plt.legend(loc = "best")
plt.show()
#Source for plotting learning curves: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

