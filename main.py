from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os, sys

training_texts = []
training_labels = []

print(os.listdir(os.path.join(os.getcwd(), "./train")))

for filename in os.listdir(os.path.join(os.getcwd(), "./train")):
    with open(os.path.join(os.getcwd(), "./train/" + filename), 'r') as f:
        linesArr = f.readlines()
        for line in linesArr:
            lineArr = line.split("	")
            for i in range(len(lineArr)):
                if i % 3 == 2:
                    training_texts.append(lineArr[i])
                if i % 3 == 1:
                    training_labels.append(lineArr[i])

testing_texts = []

sys.setrecursionlimit(1000000)

vectorizer = CountVectorizer()

vectorizer.fit(training_texts)

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(testing_texts)

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)

print(predictions)

fig = plt.figure(dpi=1200)
tree.plot_tree(classifier,feature_names = vectorizer.get_feature_names(), rounded = True, filled = True) 
fig.savefig('tree.png', dpi=1200)