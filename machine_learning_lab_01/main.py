import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('bioresponse.csv')
X = data.drop(columns=['Activity'])
y = data['Activity']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

deep_tree = DecisionTreeClassifier()
deep_tree.fit(X_train, y_train)
shallow_tree = DecisionTreeClassifier(max_depth=3)
shallow_tree.fit(X_train, y_train)

rdf_deep = RandomForestClassifier()
rdf_deep.fit(X_train, y_train)
rdf_shallow = RandomForestClassifier(max_depth=3)
rdf_shallow.fit(X_train, y_train)

pred_deep_tree = deep_tree.predict(X_test)
pred_shallow_tree = shallow_tree.predict(X_test)

pred_rdf_deep = rdf_deep.predict(X_test)
pred_rdf_shallow = rdf_shallow.predict(X_test)

#Далі йде код, який було б добре трохи відформатувати, бо купа прінтів в перемішку з іншими рядками коду.
#Плюс, задача для тебе, друже - перевірити чи збираю я всі потрібні метрики для класифікаторів(Їх тут 5).
print("deep tree: ", pred_deep_tree)
print("shallow tree: ", pred_shallow_tree)
print("y_test: ", y_test)
print(confusion_matrix(y_test, pred_deep_tree))
print(confusion_matrix(y_test, pred_shallow_tree))
print(classification_report(y_test, pred_deep_tree))
print(classification_report(y_test, pred_shallow_tree))

print("random forest with deep trees: ", pred_rdf_deep)
print("random forest with shallow trees: ", pred_rdf_shallow)
print("y_test: ", y_test)
print(confusion_matrix(y_test, pred_rdf_deep))
print(confusion_matrix(y_test, pred_rdf_shallow))
print(classification_report(y_test, pred_rdf_deep))
print(classification_report(y_test, pred_rdf_shallow))

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Отримуємо ймовірності для позитивного класу
y_prob = model.predict_proba(X_test)[:, 1]

# Встановлюємо нижчий поріг класифікації, щоб уникати помилок II роду
threshold = 0.3  # Нижчий поріг класифікації, щоб більше випадків вважати позитивними. Стандартно 0.5
y_fn_pred = (y_prob >= threshold).astype(int)

print("classifier that avoids type || errors", y_fn_pred)
print(confusion_matrix(y_test, y_fn_pred))
print(classification_report(y_test, y_fn_pred))
