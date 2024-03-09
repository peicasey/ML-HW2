import numpy as np
import xgboost as xgb

def train_XGBoost(X, y, lambdas):
    
    # Dictionary to store results
    results = {}
    
    # Perform cross-validation for each lambda value
    for lmbda in lambdas:
        # Initialize parameters for XGBoost model
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'reg_lambda': lmbda
        }
        
        # Perform cross-validation using xgb.cv
        # cv_results = xgb.cv(params, dtrain=xgb.DMatrix(X, label=y),
        #                     num_boost_round=10, nfold=10, metrics='auc',
        #                     seed=123, early_stopping_rounds=5)
        cv_results = xgb.cv(params, dtrain=xgb.DMatrix(X, label=y),
                            nfold=10, metrics='auc',
                            seed=123)
        
        # Get mean AUC score from cross-validation results
        mean_auc = cv_results['test-auc-mean'].iloc[-1]
        
        # Store mean AUC score in results dictionary
        results[lmbda] = mean_auc
        
    return results

#####################################################################################

def train_XGBoost(X, y, lambdas):
    
    results = {}
    
    # cross-validation for each lambda value
    for lmbda in lambdas:
        model = XGBClassifier(reg_lambda=lmbda)
        
        # 10-fold cross-validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=123)
        auc_scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')
        
        mean_auc = np.mean(auc_scores)
        
        results[lmbda] = mean_auc
        
    return results

