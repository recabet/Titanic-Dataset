import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model import my_LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

df.drop(["PassengerId", "Name", "Embarked", "Cabin", "Ticket", "Parch"], axis=1, inplace=True)

df.Sex = df.Sex.replace({"male": 0, "female": 1})
df = df.dropna(subset=["Age", "SibSp", "Fare", "Pclass", "Survived", "Sex"])
print(df.columns)

model = my_LogisticRegression(iterations=20000, alpha=0.001)

x = df.drop(["Survived"], axis=1)
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
ones_train = np.ones((x_train.shape[0], 1))
x_train = x_train.to_numpy()

inputs_train = np.hstack((ones_train, x_train))
y_train = np.array(y_train)

ones_test = np.ones((x_test.shape[0], 1))

x_test = x_test.to_numpy()

x_train = my_LogisticRegression.normalize(x_train)
x_test = my_LogisticRegression.normalize(x_test)

inputs_test = np.hstack((ones_test, x_test))

w = model.fit(inputs_train, y_train, cost=True)
preds = np.array(model.predict(inputs_test))
accuracy = model.accuracy(y_test, preds)
precision = model.precision(y_test, preds)
recall = model.recall(y_test, preds)
f1_score = model.f1_score(y_test, preds)


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1_score}")


conf_matrix = confusion_matrix(y_test, preds)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()


if hasattr(model, "cost_history"):
    plt.figure()
    plt.plot(model.cost_history)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()