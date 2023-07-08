# import our custom transformer
from classification_model.processing import feature as f

from sklearn.pipeline import Pipeline
#Using our final estimator to build our model
from sklearn.ensemble import RandomForestClassifier as RFC

from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
)

from feature_engine.transformation import (
    YeoJohnsonTransformer,
    LogCpTransformer
)

from feature_engine.selection import DropFeatures

from classification_model.config.core import config





credit_risk_pipeline = Pipeline([
    
    # ======== IMPUTATION ========== #

    ('missing_indicator', AddMissingIndicator(variables=config.model_config.numerical_vars_with_na)),

    ('median_imputter', MeanMedianImputer(imputation_method='median', variables=config.model_config.numerical_vars_with_na)),


    # ==== VARIABLE TRANSFORMATION ========= #

    ('log_transformer', LogCpTransformer(variables=config.model_config.numerical_log_vars, C=config.model_config.C)),

    ('yeojohnson', YeoJohnsonTransformer(variables=config.model_config.numerical_yeo_vars)),


    # ========== FEATURE EXTRACTION ========= #
    ('feature_extraction', f.DictVect(variables=config.model_config.engineered_vars)),

    # ========== SELECTION OF FEATURES SUITABLE FOR MODEL TRAINING ======= #
    ('dropped_features', DropFeatures(features_to_drop=config.model_config.dropped_vars)),


    # ======= final estimator ==========#
    ('rfc', RFC(random_state=config.model_config.random_state, 
                n_estimators= config.model_config.n_estimators, 
                max_depth=config.model_config.max_depth))

    ])