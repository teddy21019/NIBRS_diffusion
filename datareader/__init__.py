from .read_nibrs_raw import read_NIBRS_file_paths
from .BHReader import *
import pandas as pd
from pathlib import Path


def all_df_reader(pk_path: Path|str) -> pd.DataFrame:

    all_df = pd.read_pickle(pk_path)
    all_df = all_df[pd.to_numeric(all_df['NUMBER OF MONTHS REPORTED'], errors='coerce').notnull()]
    all_df["NUMBER OF MONTHS REPORTED"] = pd.to_numeric(all_df["NUMBER OF MONTHS REPORTED"])
    return all_df

def LEMAS_reader(lemas_path:Path|str) -> pd.DataFrame:
    lemas_df = pd.read_csv(lemas_path, delimiter='\t')
    return lemas_df

def _to_selected_agency_info(df:pd.DataFrame) -> pd.DataFrame:
    coords = df[["ori", "agency_name_full","longitude", "latitude"]].rename(columns={"ori":"ORI"})
    return coords

def geo_reader(csv_path:Path|str) -> pd.DataFrame:
    bs_df = pd.read_csv(csv_path)    # baseline
    geo_info = _to_selected_agency_info(bs_df)
    return geo_info