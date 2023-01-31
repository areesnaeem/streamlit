import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Title of the web app
st.title("Machine Learning Web App")
st.write('Explore differnt ML models and datasets')

# dataset selection
st.sidebar.subheader("Choose Dataset")
dataset_name = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))


# Sidebar
st.sidebar.subheader("Choose ML model")
# Dropdown to select ML model
model = st.sidebar.selectbox(
    "Select ML model", ("Random Forest", "Support Vector Machine", "KNN"))

# Function to get dataset


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


# calling function
x, y = get_dataset(dataset_name)
st.write("Shape of dataset", x.shape)
st.write("Number of classes", len(np.unique(y)))

# defining parameters for model


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Random Forest":
        n_estimators = st.sidebar.slider(
            "n_estimators", 10, 500, step=10)
        max_depth = st.sidebar.slider(
            "max_depth", 2, 15, step=1)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        n_neighbors = st.sidebar.slider(
            "n_neighbors", 1, 15, step=1)
        params["n_neighbors"] = n_neighbors
    return params


# now calling function
params = add_parameter_ui(model)

# defining function for model


def get_classifier(clf_name, params):
    if clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
    elif clf_name == "Support Vector Machine":
        clf = SVR(C=params["C"])
    else:
        clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    return clf


# calling function
clf = get_classifier(model, params)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234)

# fitting model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# only for random forest
st.write("Classifier = ", clf.__class__.__name__)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {model}")
st.write(f"Accuracy = {acc}")

# Plotting with PCA
# import pca


pca = PCA(2)
X_projected = pca.fit_transform(x)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")

plt.xlabel('Principle component 1')
plt.ylabel('Principle component 2')
plt.colorbar()
st.pyplot(fig)
