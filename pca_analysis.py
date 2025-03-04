from regression_on_year import reorder_dataframe, set_0_column

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file = "Data4_final.csv"

Data = set_0_column(pd.read_csv(file), "byg026Opførelsesår")

Principle_Components = [5,10,15,20,25]