from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
df = pd.read_csv('Movie_classification.csv')

label_mapping = {'YES': 1, 'NO': 0}

df['3D_available'] = df['3D_available'].map(label_mapping)

df = pd.get_dummies(df, columns=['Genre'], prefix=['Genre'], dtype=int)
feature_columns = list(df.columns)[:-1]

df = df.fillna(df.mean())


X = df[feature_columns]  
y = df['Start_Tech_Oscar']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy using k value of 10: {accuracy}')

neighbors = list(range(1, 50))

cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

mse = [1 - x for x in cv_scores]

optimal_k = neighbors[mse.index(min(mse))]
print(f"The optimal number of neighbors is {optimal_k}")

knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)

knn_optimal.fit(X_train, y_train)

y_pred_optimal = knn_optimal.predict(X_test)

accuracy_optimal = accuracy_score(y_test, y_pred_optimal)

print(f'Accuracy with optimal k: {accuracy_optimal}')

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


r2_score_lr = lr.score(X_test, y_test)
print(f'R^2 Score for Linear Regression: {r2_score_lr}')

logr = LogisticRegression()

logr.fit(X_train, y_train)

y_pred_logr = logr.predict(X_test)

accuracy_logr = accuracy_score(y_test, y_pred_logr)
print(f'Accuracy for Logistic Regression: {accuracy_logr}')
