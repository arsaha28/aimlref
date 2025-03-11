from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt

irisset = datasets.load_iris()
X = irisset.data
Y = irisset.target


cf = DecisionTreeClassifier(random_state=1234)
cf.fit(X,Y)
plt.figure(figsize=(20, 10))

decPlot = plot_tree(cf, 
                    feature_names=['sepal length (cm)', 
                                   'sepal width (cm)', 
                                   'petal length (cm)',
                                   'petal width (cm)'], 
                    class_names=["setosa", "versicolor", "vi"
                    "rginica"],
                    filled=True,precision=4,rounded=True)

plt.show()

text_representation = tree.export_text(cf,feature_names=['sepal length (cm)', 
                                   'sepal width (cm)', 
                                   'petal length (cm)',
                                   'petal width (cm)'])

print(text_representation)

