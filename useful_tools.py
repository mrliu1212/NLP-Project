import os
import json
import pickle
import numpy as np
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from tqdm import tqdm

class UsefulTools:
    """
    useful_tools.py
    ===============

    A general-purpose utility module that provides commonly needed tools across data science,
    machine learning, and application development projects.

    Includes:
    - JsonCache: For caching data to/from disk
    - CVGridSearch: For reusable, cacheable hyperparameter tuning via cross-validation
    """

    class JsonCache:
        @staticmethod
        def load(filename: str, expected_type: type = list):
            if not os.path.exists(filename):
                return None
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, expected_type):
                    print(f"Loaded cache from '{filename}' ({len(data)} items).")
                    return data
                else:
                    print(f"Cache file '{filename}' is invalid: expected {expected_type.__name__}.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Failed to load cache file '{filename}': {e}")
            return None

        @staticmethod
        def save(data, filename: str):
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Saved data to cache file '{filename}'.")
            except IOError as e:
                print(f"Failed to save cache to '{filename}': {e}")

        @staticmethod
        def compare_json_files(file1: str, file2: str) -> bool:
            try:
                with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
                    data1 = json.load(f1)
                    data2 = json.load(f2)
                are_equal = data1 == data2
                if are_equal:
                    print(f"The contents of '{file1}' and '{file2}' are the same.")
                else:
                    print(f"The contents of '{file1}' and '{file2}' differ.")
                return are_equal
            except Exception as e:
                print(f"Error comparing files: {e}")
                return False

    class CVGridSearch:
        def __init__(self, estimator_class, param_grid, cache_file=None, cv=3,
                     scoring='f1_macro', verbose=True, fixed_params=None):
            """
            General-purpose grid search with caching and cross-validation support.

            :param estimator_class: The classifier (e.g., LogisticRegression)
            :param param_grid: Dictionary of hyperparameters to search
            :param cache_file: Path to save/load cached results
            :param cv: Number of folds for cross-validation
            :param scoring: Scoring method (e.g., 'f1_macro', 'accuracy')
            :param verbose: Whether to print progress and results
            :param fixed_params: Any fixed parameters to pass to the model
            """
            self.estimator_class = estimator_class
            self.param_grid = list(ParameterGrid(param_grid))
            self.cache_file = cache_file
            self.cv = cv
            self.verbose = verbose
            self.scoring = scoring
            self.fixed_params = fixed_params or {}
            self.best_score = -np.inf
            self.best_params = None
            self.cache = {}

            if cache_file and os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                if verbose:
                    print(f"Loaded {len(self.cache)} cached results from {cache_file}")

        def search(self, X, y):
            for params in tqdm(self.param_grid, desc=f"{self.cv}-Fold CV"):
                key = tuple(sorted(params.items()))
                if key in self.cache:
                    score = self.cache[key]
                else:
                    model = self.estimator_class(**self.fixed_params, **params)
                    scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)
                    score = np.mean(scores)
                    self.cache[key] = score
                    if self.cache_file:
                        with open(self.cache_file, 'wb') as f:
                            pickle.dump(self.cache, f)

                if self.verbose:
                    print(f"Params: {params}, Score: {score:.4f}")

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params

            if self.verbose:
                print(f"\nBest Params: {self.best_params}")
                print(f"Best Score ({self.cv}-fold CV): {self.best_score:.4f}")

            return self.best_params, self.best_score
    
    class HalvingGridSearch:
        def __init__(self, estimator, param_grid, scoring='f1_macro', cv=3, factor=2, cache_file=None, verbose=1):
            """
            Successive halving grid search using HalvingGridSearchCV.

            :param estimator: Scikit-learn model instance (with fixed params applied)
            :param param_grid: Dictionary of parameter grid to search
            :param scoring: Scoring metric string (e.g., 'f1_macro')
            :param cv: Number of cross-validation folds
            :param factor: Halving factor (e.g., 2, 3)
            :param cache_file: Optional path to cache the fitted search object
            :param verbose: Verbosity level
            """
            self.estimator = estimator
            self.param_grid = param_grid
            self.scoring = scoring
            self.cv = cv
            self.factor = factor
            self.cache_file = cache_file
            self.verbose = verbose
            self.search = None

        def search_and_fit(self, X, y):
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.search = pickle.load(f)
                print(f"Loaded HalvingGridSearchCV from cache: {self.cache_file}")
            else:
                self.search = HalvingGridSearchCV(
                    estimator=self.estimator,
                    param_grid=self.param_grid,
                    scoring=self.scoring,
                    cv=self.cv,
                    factor=self.factor,
                    n_jobs=-1,
                    verbose=self.verbose
                )
                print("Running HalvingGridSearchCV...")
                self.search.fit(X, y)
                if self.cache_file:
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(self.search, f)
                    print(f"Saved HalvingGridSearchCV to cache: {self.cache_file}")

            print(f"Best Params: {self.search.best_params_}")
            print(f"Best Score: {self.search.best_score_:.4f}")
            return self.search.best_estimator_, self.search.best_params_, self.search.best_score_
