from time import time
from scipy.stats import randint, uniform
import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (confusion_matrix, 
                             classification_report, 
                             recall_score, f1_score, 
                             precision_score, 
                             roc_curve,
                             make_scorer)

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def scania_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = 10*fp + 500*fn
    return cost

def get_best_estimators_rf(X_train, y_train, estimators, refit_score, 
                           max_features, max_depth, min_samples_leaf, 
                           num_searches):
    
    scorers = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'scania': make_scorer(scania_score, greater_is_better=False)
    }
    
    skf = StratifiedKFold(n_splits=5)
    rf = RandomForestClassifier(n_estimators=estimators, class_weight='balanced', random_state=12, n_jobs=-1)
    
    param_dist = {'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_leaf': min_samples_leaf
                  }
    rf_rs = RandomizedSearchCV(rf, param_dist, n_iter=num_searches,
                               scoring=scorers, refit=refit_score, n_jobs=-1, 
                               cv=skf, random_state=12)
    start = time()
    rf_rs.fit(X_train, y_train)
    print('RandomizedSearchCV time: %.2f minutes' % ((time() - start) / 60))
    
    if refit_score == 'scania':
        score = -1.0 * rf_rs.best_score_
    else:
        score = rf_rs.best_score_
    best_params = rf_rs.best_params_
    best_estimator = rf_rs.best_estimator_
    print(f'---Best score: {score}')
    print(f'---Best parameters: {best_params}')
    return best_estimator 

def get_best_estimator_lgb(X_train, y_train, refit_score, num_searches):
    scorers = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'scania': make_scorer(scania_score, greater_is_better=False)
    }

    skf = StratifiedKFold(n_splits=5)
    model = lgb.LGBMClassifier(
        objective='binary', class_weight='balanced', random_state=42, n_jobs=-1)

    model_param_dist = {'boosting_type':['dart'],
                        'num_leaves':randint(30, 90),
                        'n_estimators':randint(900, 2400),
                        'learning_rate':uniform(0.01, 0.1),
                        'subsample_for_bin':randint(50000, 250000),
                        'min_child_samples':randint(50, 100),
                        'colsample_bytree':uniform(0.6, 0.3),
                        'subsample':uniform(0.4, 0.3),
                        'reg_alpha':uniform(0.0, 1.0),
                        'reg_lambda':uniform(0.0, 1.0)
                   }

    model_rs = RandomizedSearchCV(model, model_param_dist, n_iter=num_searches,
                                  scoring=scorers, refit=refit_score, n_jobs=-1,
                                  cv=skf, random_state=42)
    start = time()
    model_rs.fit(X_train, y_train)
    print('RandomizedSearchCV time: %.2f minutes' % ((time() - start) / 60))
    
    if refit_score == 'scania':
        score = -1.0 * model_rs.best_score_
    else:
        score = model_rs.best_score_
    best_params = model_rs.best_params_
    best_estimator = model_rs.best_estimator_
    # save the model to pickle
    print(f'---Best score: {score}')
    print(f'---Best parameters: {best_params}')
    return best_estimator

def get_best_thresh(model, X_test, y_test):
    """Determine the best threshold for the business problem."""
    pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)

    cost_min = np.inf
    best_threshold = 0.5
    costs = []
    for thresh in thresholds:
        y_pred_threshold = pred_proba > thresh
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        cost = 10*fp + 500*fn
        costs.append(cost)
        if cost < cost_min:
            cost_min = cost
            best_threshold = thresh
    print(f'Best thresh: {best_threshold:.4f}')
    print(f'Minimum cost: {cost_min:.2f}')
    return best_threshold

def adj_class(pred_scores, thresh):
    return [1 if score >= thresh else 0 for score in pred_scores]

def get_total_cost(model, X_test, y_test, thresh):
    """Prints confusion matrix with best threshold."""
    probs = model.predict_proba(X_test)[:, 1]
    scores_adj = adj_class(probs, thresh)
    print(confusion_matrix(y_test, scores_adj))
    tn, fp, fn, tp = confusion_matrix(y_test, scores_adj).ravel()
    cost = 10*fp + 500*fn
    print(f'\nCost of misses: {cost}')
    return cost