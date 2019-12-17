import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB

RANDOM_STATE = 61
AVERAGE_FOR_METRICS = 'macro'
MAX_ITER = 200

df = pd.read_csv('./dataset/Tweets.csv')
df = df[['tweet_id', 'text', 'airline_sentiment']]

vectorizer = CountVectorizer(stop_words='english')
vectorized_matrix = vectorizer.fit_transform(df.text)

tfid_tranformer = TfidfTransformer(smooth_idf=False)
tfid_matrix = tfid_tranformer.fit_transform(vectorized_matrix.toarray())

# Можно использовать TfidVectorizer вместо комбинации CountVectorizer и TfidfTransformer. tfid_vec_matrix == tfid_matrix
# tfid_vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=False)
# tfid_vec_matrix = tfid_vectorizer.fit_transform(df.text)

X_train, X_test, y_train, y_test = train_test_split(tfid_matrix, df[['airline_sentiment']], train_size = 0.7, random_state = RANDOM_STATE)
y_train = y_train.values.ravel()
Classifiers = [
    LogisticRegression(random_state = RANDOM_STATE, max_iter = MAX_ITER),
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    MultinomialNB(),
    ComplementNB(),
    LinearSVC(random_state = RANDOM_STATE),
    SVC(random_state = RANDOM_STATE)]
for classifier in Classifiers:
    classifier.fit(X_train.toarray(), y_train)
    prediction_train = classifier.predict(X_train.toarray())
    prediction = classifier.predict(X_test.toarray())
    print("Accuracy of " + classifier.__class__.__name__ + " on training set is " + str(accuracy_score(y_train, prediction_train)))

    print("Recall of " + classifier.__class__.__name__ + " is " + str(recall_score(y_test, prediction, average=AVERAGE_FOR_METRICS)))
    print("Precision of " + classifier.__class__.__name__ + " is " + str(precision_score(y_test, prediction, average=AVERAGE_FOR_METRICS)))
    print("F1 of " + classifier.__class__.__name__ + " is " + str(f1_score(y_test, prediction, average=AVERAGE_FOR_METRICS)))
    print(classification_report(y_test, prediction))