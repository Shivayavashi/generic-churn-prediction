from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def train_and_select_best_model(X, y):
    """Train multiple models and select the one with the highest accuracy."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = [
        ('Log', LogisticRegression(random_state=42)),
        ('Knn', KNeighborsClassifier()),
        ('Dec', DecisionTreeClassifier(random_state=42)),
        ('Ranf', RandomForestClassifier(random_state=42))
    ]
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        mean_accuracy = cv_results.mean()
        
        print(f"{name} Model Accuracy: {mean_accuracy:.4f}")
        
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_model = model
    
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model: {best_model.__class__.__name__} with accuracy: {test_accuracy:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {best_model.__class__.__name__}')
    plt.show()

    return best_model
