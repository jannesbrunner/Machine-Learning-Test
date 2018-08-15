# Using the popular Iris flower data set
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# The iris flower dataset is like the "hello world" program of datasets. 
# It's not meant to be used in practical applications, but it's good for testing machine learning techniques.

from sklearn.datasets import load_iris

iris = load_iris()
print(list(iris.target_names))
  
from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(iris.data, iris.target)

# this is clearly a setosa, all the features do match up
print(classifier.predict([[5.1,3.5,1.4,0.2]])) # = [0]

# this is ambiguous, as the last feature does not "fit" clearly
print(classifier.predict([[5.1,3.5,1.4,1.5]])) # = [0] or [1]

# The decision tree classifier randomly chooses a feature 
# that it thinks will make the best comparison which results in probabilistic behavior.



