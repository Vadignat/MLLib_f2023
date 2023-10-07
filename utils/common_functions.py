import pandas as pd
from typing import Union
import hashlib
import inspect


def read_dataframe_file(path_to_file: str) -> Union[pd.DataFrame, None]:
    if path_to_file.endswith('csv'):
        return pd.read_csv(path_to_file)
    elif path_to_file.endswith('pickle'):
        return pd.read_pickle(path_to_file)


def generate_experiment_name(base_functions: list, reg_coeff: float, lr: float) -> (str, str):
    # Convert base functions to string representation and hash them
    function_strings = [inspect.getsource(f).strip() for f in base_functions]
    concatenated = "\n".join(function_strings)
    hash_id = hashlib.md5(concatenated.encode()).hexdigest()[:6]  # taking the first 6 characters for brevity

    # Construct the name
    name = f"Reg{reg_coeff}_LR{lr}_FuncHash{hash_id}"

    return name, concatenated

