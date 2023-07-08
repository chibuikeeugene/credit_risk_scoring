# import our custom transformer
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer

# Using our final estimator to build our model
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import Pipeline

from classification_model.config.core import config
from classification_model.processing import feature as f

credit_risk_pipeline = Pipeline(
    [
        # ======== IMPUTATION ========== #
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
        ),
        (
            "median_imputter",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        # ==== VARIABLE TRANSFORMATION ========= #
        (
            "logcp_transformer",
            LogCpTransformer(
                variables=config.model_config.numerical_log_vars,
                C=config.model_config.C,
            ),
        ),
        (
            "yeojohnson",
            YeoJohnsonTransformer(variables=config.model_config.numerical_yeo_vars),
        ),
        # ========== FEATURE EXTRACTION ========= #
        (
            "feature_extraction",
            f.DictVect(variables=config.model_config.engineered_vars),
        ),
        # ========== SELECTION OF FEATURES SUITABLE FOR MODEL TRAINING ======= #
        (
            "dropped_features",
            DropFeatures(features_to_drop=config.model_config.dropped_vars),
        ),
        # ======= final estimator ==========#
        (
            "rfc",
            RFC(
                random_state=config.model_config.random_state,
                n_estimators=config.model_config.n_estimators,
                max_depth=config.model_config.max_depth,
            ),
        ),
    ]
)
