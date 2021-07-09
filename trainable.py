import os

import numpy as np
from sklearn.base import clone
from sklearn.model_selection._validation import (
    indexable,
    check_cv,
    is_classifier,
    check_scoring,
    _check_multimetric_scoring,
    _insert_error_scores,
    _normalize_score_results,
    _aggregate_score_dicts,
    _fit_and_score,
)

import ray
from ray import tune


@ray.remote
def ray_fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    return _fit_and_score(
        estimator=estimator,
        X=X,
        y=y,
        scorer=scorer,
        train=train,
        test=test,
        verbose=verbose,
        parameters=parameters,
        fit_params=fit_params,
        return_train_score=return_train_score,
        return_parameters=return_parameters,
        return_n_test_samples=return_n_test_samples,
        return_times=return_times,
        return_estimator=return_estimator,
        split_progress=split_progress,
        candidate_progress=candidate_progress,
        error_score=error_score,
    )


class BenchmarkTrainable(tune.Trainable):
    def setup(self, config, X=None, y=None, cv=4, estimator=None):
        self.config = config
        self.X_ref = X[0]
        self.y_ref = y[0]
        self.X = ray.get(self.X_ref)
        self.y = ray.get(self.y_ref)
        self.cv = cv
        self.estimator = estimator

    def step(self):
        config = self.config.copy()
        config["n_estimators"] = int(round(config["n_estimators"]))
        config["num_leaves"] = int(round(config["num_leaves"]))
        config["min_child_samples"] = int(round(config["min_child_samples"]))
        config["max_bin"] = 1 << int(round(config["log_max_bin"])) - 1
        estimator = clone(self.estimator).set_params(**config)
        cv_results = self._cross_validate(
            estimator,
            X=self.X,
            X_ref=self.X_ref,
            y=self.y,
            y_ref=self.y_ref,
            scoring="roc_auc"
            if len(np.unique(self.y)) == 2 else "roc_auc_ovr_weighted",
            cv=self.cv,
        )
        return {"roc_auc": np.mean(cv_results["test_score"])}

    def reset_config(self, new_config):
        self.config = new_config
        return True

    def _cross_validate(
        self,
        estimator,
        X,
        X_ref,
        y=None,
        y_ref=None,
        *,
        groups=None,
        scoring=None,
        cv=None,
        verbose=0,
        fit_params=None,
        return_train_score=False,
        return_estimator=False,
        error_score=np.nan,
    ):
        """Fast cross validation with Ray, adapted from sklearn.validation.cross_validate"""
        X, y, groups = indexable(X, y, groups)

        cv = check_cv(cv, y, classifier=is_classifier(estimator))

        if callable(scoring):
            scorers = scoring
        elif scoring is None or isinstance(scoring, str):
            scorers = check_scoring(estimator, scoring)
        else:
            scorers = _check_multimetric_scoring(estimator, scoring)

        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.
        train_test = list(cv.split(X, y, groups))

        results_futures = [
            ray_fit_and_score.remote(
                clone(estimator),
                X_ref,
                y_ref,
                scorers,
                train,
                test,
                verbose,
                None,
                fit_params,
                return_train_score=return_train_score,
                return_times=True,
                return_estimator=return_estimator,
                error_score=error_score,
            ) for train, test in train_test
        ]

        results = ray.get(results_futures)

        # For callabe scoring, the return type is only know after calling. If the
        # return type is a dictionary, the error scores can now be inserted with
        # the correct key.
        if callable(scoring):
            _insert_error_scores(results, error_score)

        results = _aggregate_score_dicts(results)

        ret = {}
        ret["fit_time"] = results["fit_time"]
        ret["score_time"] = results["score_time"]

        if return_estimator:
            ret["estimator"] = results["estimator"]

        test_scores_dict = _normalize_score_results(results["test_scores"])
        if return_train_score:
            train_scores_dict = _normalize_score_results(
                results["train_scores"])

        for name in test_scores_dict:
            ret["test_%s" % name] = test_scores_dict[name]
            if return_train_score:
                key = "train_%s" % name
                ret[key] = train_scores_dict[name]

        return ret