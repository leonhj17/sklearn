# _*_ encoding:utf-8 _*_

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import graphviz

wine = load_wine()
features = wine.feature_names
# wine_frame = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# wine_frame.to_pickle('wine_frame.pkl')
# wine_frame = pd.read_pickle('wine_frame.pkl')

xtrain, xtest, ytrain, ytest = train_test_split(wine.data, wine.target, test_size=0.3)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(xtrain, ytrain)
score = clf.score(xtest, ytest)

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=features,
    class_names = wine.target_names,
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data)
graph.view()

# clf.feature_importances_







