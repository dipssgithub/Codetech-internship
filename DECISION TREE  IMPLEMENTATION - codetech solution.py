#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = load_iris()
X = data.data  
y = data.target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names.tolist(),  # Convert array to list
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()


# In[ ]:




