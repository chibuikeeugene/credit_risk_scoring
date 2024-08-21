import os
from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import YAML, load


# Project Directories
PACKAGE_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """application level config"""

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """all configuration relevant to model training and feature engineering."""

    target: str
    features: List[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int
    C: int
    numerical_vars_with_na: List[str]
    numerical_log_vars: List[str]
    numerical_yeo_vars: str
    categorical_vars: List[str]


class Config(BaseModel):
    """master config object"""

    app_config: AppConfig
    models_config: ModelConfig


def find_config_file() -> Path:
    """locate the configuration file"""
    if os.path.isfile(CONFIG_FILE_PATH):
        return CONFIG_FILE_PATH 
    raise Exception(f"config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """parse the YAML containing the package configuration"""
    if cfg_path:
        cfg_path = find_config_file()
        with open(cfg_path, "rb") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """run  validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        models_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
print(config.model_config.numerical_vars_with_na)