import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:

    df = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))

    # format the column header case
    df.columns = df.columns.str.lower()

    # update the categorical var its string values so we can know what each number represent
    status_values = {1: "good", 2: "bad", 0: "unknown"}
    df.status = df.status.map(status_values)

    home_values = {
        1: "rent",
        2: "owner",
        3: "priv",
        4: "ignore",
        5: "parents",
        6: "other",
        0: "unknown",
    }
    df.home = df.home.map(home_values)

    marital_values = {
        1: "single",
        2: "married",
        3: "widow",
        4: "separated",
        5: "divorced",
        0: "unknown",
    }
    df.marital = df.marital.map(marital_values)

    records_values = {1: "no_rec", 2: "yes_rec"}
    df.records = df.records.map(records_values)

    job_values = {1: "fixed", 2: "partime", 3: "freelance", 4: "others", 0: 'unknown"'}
    df.job = df.job.map(job_values)

    # 99999999 represents data not available for a particular user. Hence, let's
    # replace them with the usual NaN in numoy

    num_List = ["income", "assets", "debt"]
    for var in num_List:
        df[var].replace(to_replace=99999999, value=np.nan, inplace=True)

    # let's exclude the unknown value in status since their present is small
    df = df[df.status != "unknown"]

    # let's change the status value from string data type to int.
    df.status = (df.status == "good").astype(int)

    return df


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # prepare the versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
