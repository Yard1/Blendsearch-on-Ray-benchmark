import argparse
import numpy as np
import functools
import asyncio
from unittest import mock

import ray
import gc
from ray import tune
from ray.exceptions import RayActorError
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.optuna import OptunaSearch
from flaml.searcher import BlendSearch, CFO
from ray.tune import with_parameters, PlacementGroupFactory
from ray.tune.trial_runner import TrialRunner, Trial

from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing

import openml

import numpy as np
from sklearn.base import clone
from sklearn.model_selection._validation import (
    indexable,
    check_cv,
    is_classifier,
    check_scoring,
    _check_multimetric_scoring,
    _fit_and_score,
)

import ray
from ray import tune

tasks = {
    41169: "helena",
    41168: "jannis",
    41166: "volkert",
    41165: "robert",
    41161: "riccardo",
    41159: "guillermo",
    41150: "MiniBooNE",
    41138: "APSFailure",
    41027: "jungle_chess_2pcs_raw_endgame_complete",
    40996: "Fashion-MNIST",
    40685: "shuttle",
    40668: "connect-4",
    23517: "numerai28.6",
    23512: "higgs",
    4135: "Amazon_employee_access",
    1590: "adult",
    1486: "nomao",
    1461: "bank-marketing",
    1111: "KDDCup09_appetency",
}

# searchers = ["optuna", "cfo"]
# searchers = ["random", "optuna", "blendsearch", "cfo"]
searchers = ["optuna", "blendsearch"]


class _EventActor:
    def __init__(self):
        self._event = asyncio.Event()
        self._actors = []

    def set(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def add_actor(self, actor):
        self._actors.extend(actor)
        return True

    def is_set(self):
        return self._event.is_set()


class Event:
    def __init__(self, actor_options=None):
        actor_options = {} if not actor_options else actor_options
        self.actor = ray.remote(_EventActor).options(**actor_options).remote()

    def set(self):
        ray.get(self.actor.set.remote())

    def clear(self):
        self.actor.clear.remote()

    def is_set(self):
        return ray.get(self.actor.is_set.remote())

    def add_actor(self, actor):
        ray.get(self.actor.add_actor.remote(actor))

    def shutdown(self):
        if self.actor:
            ray.kill(self.actor)
        self.actor = None


def get_current_node_resource_key() -> str:
    """Get the Ray resource key for current node.
    It can be used for actor placement.
    If using Ray Client, this will return the resource key for the node that
    is running the client server.
    """
    current_node_id = ray.get_runtime_context().node_id.hex()
    for node in ray.nodes():
        if node["NodeID"] == current_node_id:
            # Found the node.
            for key in node["Resources"].keys():
                if key.startswith("node:"):
                    return key
    else:
        raise ValueError("Cannot found the node dictionary for current node.")


def force_on_current_node(task_or_actor):
    """Given a task or actor, place it on the current node.

    If the task or actor that is passed in already has custom resource
    requirements, then they will be overridden.

    If using Ray Client, the current node is the client server node.
    """
    node_resource_key = get_current_node_resource_key()
    print(node_resource_key)
    options = {"resources": {node_resource_key: 0.01}}
    return task_or_actor.options(**options)


def col_to_fp32(col):
    if col.dtype.name == "float64":
        return col.astype("float32")
    return col


def get_dataset(id):
    if id not in tasks:
        raise ValueError(
            f"Wrong dataset id. Can be one of: {list(tasks.keys())}")
    name = tasks[id]
    print(f"Downloading dataset {name} ({id})")
    dataset = openml.datasets.get_dataset(id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X = X.apply(col_to_fp32)
    print(f"Dataset {name} ({id}) downloaded")
    return (name, X, y)


def score_on_test(estimator: LGBMClassifier, config, X_train, y_train, X_test,
                  y_test, **kwargs):
    config = config.copy()
    estimator = clone(estimator)
    config["n_estimators"] = int(round(config["n_estimators"]))
    config["num_leaves"] = int(round(config["num_leaves"]))
    config["min_child_samples"] = int(round(config["min_child_samples"]))
    config["max_bin"] = 1 << int(round(config.pop("log_max_bin"))) - 1
    estimator.set_params(**config)
    estimator.set_params(callbacks=None)
    estimator.fit(X_train, y_train, **kwargs)
    y_proba = estimator.predict_proba(X_test)
    if len(np.unique(y_test)) == 2:
        y_proba = y_proba[:, 1]
    return roc_auc_score(y_test,
                         y_proba,
                         average="weighted",
                         multi_class="ovr")


@ray.remote
def ray_score_on_test(estimator,
                      config,
                      X_train,
                      y_train,
                      X_test,
                      y_test,
                      kwargs=None):
    kwargs = kwargs or {}
    try:
        return score_on_test(estimator, config, X_train, y_train, X_test,
                             y_test, **kwargs)
    except (ValueError, SystemExit):
        return np.nan


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
    def setup(self, config, folds, estimator=None, stop_event=None):
        self.config = config
        self.folds = folds
        self.estimator = estimator
        self.actors = []
        self._stop_event = stop_event[0]
        self.stop_event = ray.get(stop_event[0])

    def _stop_callback(self):
        class callback:
            def __init__(self, stop_event):
                self.stop_event = stop_event
                self.stop_event_obtained = False

            def __call__(self, env) -> None:
                if not self.stop_event_obtained:
                    self.stop_event = ray.get(self.stop_event)
                    self.stop_event_obtained = True
                try:
                    if self.stop_event.is_set():
                        print("stop event callback triggered")
                        raise ValueError()
                except RayActorError:
                    print("stop event callback triggered")
                    raise ValueError()

            self.order = 1  # type: ignore

        return callback(self._stop_event)

    def step(self):
        config = self.config.copy()
        self.actors = [
            ray_score_on_test.remote(self.estimator, config, X_train, y_train,
                                     X_test, y_test)
            for X_train, y_train, X_test, y_test in self.folds
        ]
        remaining_actors = self.actors
        self.stop_event.add_actor(remaining_actors)
        while remaining_actors:
            if self.stop_event.is_set():
                print("stop event is set, cleaning up")
                return {"roc_auc": np.nan}
            results, remaining_actors = ray.wait(remaining_actors, timeout=0.5)
        return {"roc_auc": np.mean(ray.get(results))}

    def reset_config(self, new_config):
        self.config = new_config
        return True


def get_cv_folds(
    estimator,
    X,
    y=None,
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

    return train_test


def benchmark(searcher,
              dataset,
              seed,
              max_concurrent=8,
              cv=3,
              time_budget_s=15 * 60,
              test_size=0.3):
    gc.collect()
    dataset_name, X, y, = dataset
    print(
        f"Starting benchmark on dataset {dataset_name}, seed:{seed}, max_concurrent:{max_concurrent}, cv:{cv}, time_budget_s:{time_budget_s}"
    )
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=seed,
                                                        stratify=y)
    upper = min(32768, int(len(X_train) // cv))

    intlog = functools.partial(tune.qloguniform,
                               q=1) if searcher == "cfo" else tune.lograndint
    lgbm_config = {
        "n_estimators": intlog(lower=4, upper=upper),
        "num_leaves": intlog(lower=4, upper=upper),
        "min_child_samples": intlog(lower=2, upper=2**7),
        "log_max_bin": intlog(lower=3, upper=10),
        "subsample": tune.uniform(lower=0.1, upper=1.0),
        "colsample_bytree": tune.uniform(lower=0.01, upper=1.0),
        "reg_alpha": tune.loguniform(lower=1 / 1024, upper=10.0),
        "reg_lambda": tune.loguniform(lower=1 / 1024, upper=10.0),
    }
    init_config = {
        "n_estimators": 100,
        "num_leaves": 31,
        "min_child_samples": 20,
        "log_max_bin": 8,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 1 / 1024,
        "reg_lambda": 1 / 1024,
    }
    blendsearch_low_cost_config = {
        "n_estimators": 4,
        "num_leaves": 4,
    }

    if searcher == "random":
        search_alg = BasicVariantGenerator(max_concurrent=max_concurrent)
    elif searcher == "optuna":
        search_alg = OptunaSearch(sampler=TPESampler(multivariate=True,
                                                     seed=seed),
                                  seed=seed,
                                  points_to_evaluate=[init_config])
    elif searcher == "blendsearch":
        search_alg = BlendSearch(
            seed=seed,
            metric="roc_auc",
            mode="max",
            space=lgbm_config,
            points_to_evaluate=[init_config],
            low_cost_partial_config=blendsearch_low_cost_config)
        search_alg.set_search_properties(config={"time_budget_s": time_budget_s})
    elif searcher == "cfo":
        search_alg = CFO(seed=seed,
                         points_to_evaluate=[init_config],
                         low_cost_partial_config=blendsearch_low_cost_config)
    else:
        raise ValueError(f"Wrong searcher. Can be one of {searchers}")

    if searcher != "random":
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent)

    estimator = LGBMClassifier(random_state=seed, n_jobs=1)

    cv_folds = get_cv_folds(estimator, X, y, cv=cv)
    folds = [(ray.put(_safe_indexing(X,
                                     train)), ray.put(_safe_indexing(y,
                                                                     train)),
              ray.put(_safe_indexing(X,
                                     test)), ray.put(_safe_indexing(y, test)))
             for train, test in cv_folds]

    stop_event = Event()

    trainable_with_parameters = with_parameters(
        BenchmarkTrainable,
        folds=folds,
        estimator=estimator,
        stop_event=(ray.put(stop_event), ))

    name = f"benchmark_{dataset_name}_{searcher}_{seed}"

    class StoppingTrialRunner(TrialRunner):
        _stop_event = stop_event

        def _stop_experiment_if_needed(self):
            fail_fast = self._fail_fast and self._has_errored
            if (self._stopper.stop_all() or fail_fast
                    or self._should_stop_experiment):
                self._search_alg.set_finished()
                print("setting stop event")
                self._stop_event.set()
                [
                    self.trial_executor.stop_trial(t) for t in self._trials
                    if t.status is not Trial.ERROR
                ]

    gc.collect()
    print(f"Starting tune.run")
    with mock.patch("ray.tune.tune.TrialRunner", StoppingTrialRunner):
        analysis = tune.run(trainable_with_parameters,
                            config=lgbm_config,
                            search_alg=search_alg,
                            name=name,
                            num_samples=-1,
                            time_budget_s=time_budget_s,
                            metric="roc_auc",
                            mode="max",
                            verbose=2,
                            stop={"training_iteration": 1},
                            resources_per_trial=PlacementGroupFactory([{
                                "CPU": 1
                            }] * cv),
                            max_failures=2,
                            reuse_actors=False,
                            resume=False,
                            raise_on_failed_trial=False)

    test_result = ray_score_on_test.remote(estimator, analysis.best_config,
                                           X_train, y_train, X_test, y_test)
    test_result = ray.get(test_result)
    print(f"Test result: {test_result}")

    results = analysis.results_df.copy()
    results.to_csv(f"{name}.csv")
    with open(f"{name}.test_result", "w") as f:
        f.write(str(test_result))
    return test_result, results, name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("max_concurrent",
                        type=int,
                        help="Max concurrent trials")
    parser.add_argument("time_budget_s",
                        type=float,
                        help="Time budget in seconds")
    parser.add_argument(
        "--dataset",
        required=False,
        type=int,
        default=None,
        help=
        "Dataset id to use. If not specified will run all predefined datasets")
    parser.add_argument("--searcher",
                        required=False,
                        type=str,
                        default=None,
                        help="Searcher to use")
    parser.add_argument("--cv",
                        required=False,
                        type=int,
                        default=3,
                        help="Number of cv folds")
    parser.add_argument("--test_size",
                        required=False,
                        type=float,
                        default=0.3,
                        help="Fraction of dataset to be held back for testing")
    parser.add_argument("--address",
                        required=False,
                        type=str,
                        default=None,
                        help="the address to use for Ray")
    parser.add_argument("--server-address",
                        type=str,
                        default=None,
                        required=False,
                        help="The address of server to connect to if using "
                        "Ray Client.")
    args, _ = parser.parse_known_args()
    if args.server_address:
        ray.util.connect(args.server_address)
    else:
        ray.init(address=args.address)

    if args.dataset is None:
        for id in tasks:
            dataset = get_dataset(id)
            if args.searcher is None:
                for searcher in searchers:
                    benchmark(searcher,
                              dataset,
                              args.seed,
                              max_concurrent=args.max_concurrent,
                              time_budget_s=args.time_budget_s,
                              cv=args.cv)
            else:
                benchmark(args.searcher,
                          dataset,
                          args.seed,
                          max_concurrent=args.max_concurrent,
                          time_budget_s=args.time_budget_s,
                          cv=args.cv)
    else:
        dataset = get_dataset(args.dataset)
        if args.searcher is None:
            for searcher in searchers:
                benchmark(searcher,
                          dataset,
                          args.seed,
                          max_concurrent=args.max_concurrent,
                          time_budget_s=args.time_budget_s,
                          cv=args.cv)
        else:
            benchmark(args.searcher,
                      dataset,
                      args.seed,
                      max_concurrent=args.max_concurrent,
                      time_budget_s=args.time_budget_s,
                      cv=args.cv)
