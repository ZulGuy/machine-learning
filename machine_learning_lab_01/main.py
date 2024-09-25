import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, log_loss


def print_res(message, func):
    color_blue = "\033[94m"
    color_reset = "\033[0m"
    formatted_message = f"{color_blue}{message}\n{color_reset}{func}"
    print(formatted_message)


# Data Loading and Preprocessing
data = pd.read_csv('bioresponse.csv')
X = data.drop(columns=['Activity'])
y = data['Activity']

# Split data into training and testing sets using an 80-20 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

# Training Decision Tree Models
deep_tree = DecisionTreeClassifier()  # A deep decision tree with no max depth
deep_tree.fit(X_train, y_train)
shallow_tree = DecisionTreeClassifier(max_depth=3)  # A shallow decision tree with a maximum depth of 3
shallow_tree.fit(X_train, y_train)

# Training Random Forest Models:
rdf_deep = RandomForestClassifier()
rdf_deep.fit(X_train, y_train)
rdf_shallow = RandomForestClassifier(max_depth=3)
rdf_shallow.fit(X_train, y_train)

# Making Predictions:
pred_deep_tree = deep_tree.predict(X_test)
pred_shallow_tree = shallow_tree.predict(X_test)

pred_rdf_deep = rdf_deep.predict(X_test)
pred_rdf_shallow = rdf_shallow.predict(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Getting probabilities for the positive class
y_prob = model.predict_proba(X_test)[:, 1]

# Setting a lower classification threshold to avoid Type II errors
threshold = 0.3  # Lower classification threshold to consider more cases as positive. Default is 0.5
y_fn_pred = (y_prob >= threshold).astype(int)

# List of models
models = {
    "Deep Decision Tree": deep_tree,
    "Shallow Decision Tree": shallow_tree,
    "Random Deep Forest": rdf_deep,
    "Random Shallow Forest": rdf_shallow,
    "Classifier that avoids 2nd type errors": model
}

# Prepare the plot
plt.figure(figsize=(10, 8))

# Building ROC curves for each model
for model_name, model in models.items():

    # Probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    # Building the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Adding the curve to the plot
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot parameters
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for different models')
plt.legend(loc="lower right")

# Display the plot
plt.show()

print_res("Deep tree:", pred_deep_tree)
print_res("Shallow tree: ", pred_shallow_tree)
print_res("y_test: ", y_test)
print_res("Confusion matrix based on pred_deep_tree:", confusion_matrix(y_test, pred_deep_tree))
print_res("Confusion matrix based on pred_shallow_tree:", confusion_matrix(y_test, pred_shallow_tree))
print_res("Classification report based on pred_deep_tree:", classification_report(y_test, pred_deep_tree))
print_res("Classification report based on pred_shallow_tree:", (classification_report(y_test, pred_shallow_tree)))

print_res("Random forest with deep trees: ", pred_rdf_deep)
print_res("Random forest with shallow trees: ", pred_rdf_shallow)
print_res("y_test: ", y_test)
print_res("Confusion matrix based on pred_rdf_deep:", confusion_matrix(y_test, pred_rdf_deep))
print_res("Confusion matrix based on pred_rdf_shallow:", confusion_matrix(y_test, pred_rdf_shallow))
print_res("Classification report based on pred_rdf_deep:", classification_report(y_test, pred_rdf_deep))
print_res("Classification report based on pred_rdf_shallow:", classification_report(y_test, pred_rdf_shallow))

print_res("classifier that avoids type || errors", y_fn_pred)
print_res("Confusion matrix based on y_fn_pred:", confusion_matrix(y_test, y_fn_pred))
print_res("Classification report based on y_fn_pred:", classification_report(y_test, y_fn_pred))

# Calculate log_loss for each model
for model_name, model in models.items():

    # Get probabilities for the positive class
    y_proba = model.predict_proba(X_test)

    # Calculate log_loss
    loss = log_loss(y_test, y_proba)

    print(f'{model_name} log loss: {loss:.4f}')
