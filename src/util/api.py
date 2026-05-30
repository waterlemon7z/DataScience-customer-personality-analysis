from pathlib import Path

import kagglehub
import pandas as pd


def load_dataset_from_kaggle(
        dataset: str = "imakash3011/customer-personality-analysis",
        file_name: str = "marketing_campaign.csv",
) -> pd.DataFrame:
    """
    - Name - load_dataset_from_kaggle
    - Type - Function
    - Params - dataset: str, file_name: str
    - Returns - pd.DataFrame
    - Description - Get csv data from kaggle server.
    """
    path = Path(kagglehub.dataset_download(dataset))
    return pd.read_csv(path / file_name, sep="\t").copy()