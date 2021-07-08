import argparse
import numpy as np
import functools

import ray
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.flaml import BlendSearch
from ray.tune import with_parameters, PlacementGroupFactory

from optuna.samplers import TPESampler
from lightgbm import LGBMClassifier

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import openml

from trainable import BenchmarkTrainable

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

searchers = ["random", "optuna", "blendsearch"]


def col_to_fp32(col):
    if col.dtype.name == "float64":
        return col.astype("float32")
    return col


def score_on_test(estimator: LGBMClassifier, config, X_train, y_train, X_test,
                  y_test):
    config = config.copy()
    estimator = clone(estimator)
    config["n_estimators"] = int(config["n_estimators"])
    config["num_leaves"] = int(config["num_leaves"])
    config["min_child_samples"] = int(config["min_child_samples"])
    estimator.set_params(**config)
    estimator.fit(X_train, y_train)
    y_proba = estimator.predict_proba(X_test)
    if len(np.unique(y_test)) == 2:
        y_proba = y_proba[:, 1]
    return roc_auc_score(y_test,
                         y_proba,
                         average="weighted",
                         multi_class="ovr")


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


def benchmark(searcher,
              dataset,
              seed,
              max_concurrent=8,
              cv=3,
              time_budget_s=15 * 60,
              test_size=0.3):
    dataset_name, X, y, = dataset
    print(
        f"Starting benchmark on dataset {dataset_name}, seed:{seed}, max_concurrent:{max_concurrent}, cv:{cv}, time_budget_s:{time_budget_s}"
    )
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=seed,
                                                        stratify=y)
    upper = min(2048, int(len(X_train) // cv))

    intlog = functools(tune.qloguniform,
                       q=1) if searcher == "blendsearch" else tune.lograndint
    lgbm_config = {
        "n_estimators": intlog(lower=4, upper=upper),
        "num_leaves": intlog(lower=4, upper=255),
        "min_child_samples": intlog(lower=2, upper=2**7),
        "subsample": tune.uniform(lower=0.1, upper=1.0),
        "colsample_bytree": tune.uniform(lower=0.01, upper=1.0),
        "reg_alpha": tune.loguniform(lower=1 / 1024, upper=10.0),
        "reg_lambda": tune.loguniform(lower=1 / 1024, upper=10.0),
    }
    init_config = {
        "n_estimators": 100,
        "num_leaves": 31,
        "min_child_samples": 20,
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
            points_to_evaluate=[init_config],
            low_cost_partial_config=blendsearch_low_cost_config)
    else:
        raise ValueError(f"Wrong searcher. Can be one of {searchers}")

    if searcher != "random":
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent)

    estimator = LGBMClassifier(random_state=seed, n_jobs=1)

    trainable_with_parameters = with_parameters(BenchmarkTrainable,
                                                X=(ray.put(X_train), ),
                                                y=(ray.put(y_train), ),
                                                cv=cv,
                                                estimator=estimator)

    name = f"benchmark_{dataset_name}_{searcher}_{seed}"

    print(f"Starting tune.run")
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
                        fail_fast=True,
                        reuse_actors=True)

    print(f"Scoring on {test_size} of the dataset")
    test_result = score_on_test(estimator, analysis.best_config, X_train,
                                y_train, X_test, y_test)
    print(f"Test result: {test_result}")

    print(f"Saving to '{name}'")
    results = analysis.results_df.copy()
    results.to_csv(f"{name}.csv")
    with open(f"{name}.test_result", "w") as f:
        f.write(str(test_result))
    return analysis


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
        ray.init(address=args.address, num_cpus=8)

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
