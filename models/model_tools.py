#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split,validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings("ignore")



def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    
    knn_params = {"n_neighbors": range(2, 5)}

    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    rf_params = {"max_depth": [4, 5, 6, 7],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}

    xgboost_params = {"learning_rate": [0.1, 0.01, 0.03, 0.3],
                      "max_depth": [4, 5, 6, 7],
                      "n_estimators": [100, 200, 300]}

    lightgbm_params = {"learning_rate": [0.01, 0.1, 0.03, 0.3],
                       "n_estimators": [300, 500]}


    classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                   ("CART", DecisionTreeClassifier(), cart_params),
                   ("RF", RandomForestClassifier(), rf_params),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
                   ('LightGBM', LGBMClassifier(), lightgbm_params)]
    print("Hyperparameter Optimization....")
    best_models = {}
        
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def cosine_predictor(X_train, X_test, y_train):
    # compute cosine similarity between train and test data
    similarity = cosine_similarity(X_test, X_train)

    # find most similar training row for each test row
    most_similar_index = np.argmax(similarity, axis=1)

    # predict target using most similar training row
    y_pred = y_train.iloc[most_similar_index].values
    
    return y_pred

def pca_analysis(data, n_components=None, plot=False):
    """
    Principal Component Analysis (PCA) testi

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Gözlemlerin ve özelliklerin olduğu veri kümesi.

    n_components : int or None (default=None)
        Öznitelik sayısı. Varsayılan değer None, tüm öznitelikleri korur.

    plot : bool (default=False)
        PCA sonuçlarının görselleştirilip görselleştirilmeyeceğini belirler.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        PCA objesi.

    eig_values : ndarray, shape (n_components,)
        Elde edilen özdeğerlerin listesi.

    cum_var_ratio : ndarray, shape (n_components,)
        Elde edilen kümülatif varyans oranlarının listesi.

    """
    # PCA modeli oluştur
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # Elde edilen öznitelik sayısını belirle
    if n_components is None:
        n_components = pca.n_components_

    # Elde edilen öznitelik sayısına göre PCA sonuçlarını hesapla
    eig_values = pca.explained_variance_[:n_components]
    cum_var_ratio = np.cumsum(pca.explained_variance_ratio_[:n_components])

    # Elbow Grafiği
    plt.plot(range(1, n_components + 1), eig_values, marker='o', color='green')
    plt.title('Elbow Grafiği')
    plt.xlabel('Bileşen Numarası')
    plt.ylabel('Özdeğer')
    plt.show()

    # PCA Sonuçları Görselleştirme
    if plot:
        pca_components = pca.transform(data)[:, :n_components]

        # Renkli Görselleştirme
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_components[:, 0], pca_components[:, 1], pca_components[:, 2], c=data[:, 0], cmap='viridis')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()

    return pca, eig_values, cum_var_ratio
#kullanim sekli:pca, eig_values, cum_var_ratio = pca_analysis(df, n_components=3, plot=False)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
    '''--Eğer eğitim skoru (mavi çizgi) yüksek ve doğrulama skoru (yeşil çizgi) düşükse, 
    --bu genellikle aşırı uyum (overfitting) sorununa işaret eder. Bu durumda, model eğitim 
        --verilerine aşırı uyum sağlamış ve genelleme yaparken daha düşük bir performans sergilemiştir.

    --Eğer eğitim skoru (mavi çizgi) ve doğrulama skoru (yeşil çizgi) düşükse, 
        --bu genellikle aşırı basitlik (underfitting) sorununa işaret eder. Bu durumda, model eğitim verilerine 
            --yeterince uyum sağlayamamış ve daha karmaşık bir modelin gerektiği anlamına gelebilir.

    --Eğer eğitim skoru (mavi çizgi) ve doğrulama skoru (yeşil çizgi) yüksek 
        --ve yakın değerlerdeyse, bu genellikle uygun bir model karmaşıklığı seçildiğine işaret eder. 
            --Bu durumda, model eğitim verilerine uygun uyum sağlamış ve aynı zamanda yeni verilerde de iyi bir performans 
                --sergileyecek şekilde genelleme yapabilir.'''
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

def plot_learning_curve(estimator, X, y, train_sizes = np.linspace(0.1, 1.0, 10), cv=5, scoring="accuracy"):
    """
    Plots a learning curve for an estimator using a training set of varying sizes.

    Parameters:
    -----------
    estimator : sklearn estimator object
        The estimator to use for the learning curve.

    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target values.

    train_sizes : array-like of shape (n_ticks,)
        The training set sizes to use for the learning curve.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        None, to use the default 5-fold cross-validation.
        
    scoring : string, callable, list/tuple or dict, optional
        A scoring metric to evaluate predictions on the test set.
        None, to use the default estimator score method.

    Returns:
    --------
    learning_curve : matplotlib.pyplot object
        The learning curve plot.
    """

    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            train_sizes=train_sizes,
                                                            cv=cv,
                                                            scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt.show()

def plot_confusion_matrix(y_true, y_pred, num_labels):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create heatmap of confusion matrix
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=np.arange(num_labels), yticklabels=np.arange(num_labels))
    
    # Add axis labels and title
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    
    # Show the plot
    plt.show()

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

