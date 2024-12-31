from skopt import BayesSearchCV
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV

# Set the seed for reproducibility
seed = 123

# Define stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

# Define hyperparameter tuning methods
# Random Search Tuning
def random_search_tuning(model, param_dist, X_train, y_train):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20,
                                       scoring=scorer, cv=skf, random_state=seed,
                                       n_jobs=-1, error_score='raise')
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Grid Search Tuning
def grid_search_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scorer, cv=skf,
                               n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Bayesian Optimization Tuning
def bayesian_optimization_tuning(model, search_spaces, X_train, y_train):
    bayes_search = BayesSearchCV(model, search_spaces=search_spaces, n_iter=10,
                                 scoring=scorer, cv=skf, random_state=seed,
                                 n_jobs=-1, error_score='raise')
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_estimator_


# Initialize models
catboost_model = CatBoostClassifier(verbose=0, random_seed=seed)
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed)
lightgbm_model = LGBMClassifier(random_state=seed)

# Apply SMOTE in a pipeline to handle imbalance
smote = SMOTE(random_state=seed)

catboost_pipeline = Pipeline([('smote', smote), ('classifier', catboost_model)])
xgboost_pipeline = Pipeline([('smote', smote), ('classifier', xgboost_model)])
lightgbm_pipeline = Pipeline([('smote', smote), ('classifier', lightgbm_model)])
