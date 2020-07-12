from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from processing import preprocessors as pp
from config import config


interest_rate_pipe = Pipeline(
    [
        (
            'numerical_imputer',
            pp.NumericalImputer(features=config.NUM_VARS_WITH_NA),
        ),
        (
            'random_forest',
            RandomForestRegressor(
                n_estimators=100,
                max_depth=18,
                max_features=0.9,
                min_samples_leaf=3,
                random_state=3,
                n_jobs=-1)
        )
    ]
)
