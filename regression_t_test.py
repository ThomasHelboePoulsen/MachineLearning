import pandas as pd
from scipy.stats import ttest_rel
import numpy as np

def main():
    df = pd.read_excel("tTestResults.xlsx")
    differenceTest(df["ann"],df["linreg"])

# bootstrap assuming no distribution
def bootstrap10(data):
        n_bootstrap = 10 * len(data)
        bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
        return bootstrap_samples

def differenceTest(vector1,vector2):
    """get difference measures. Must be dependent"""
    t_stat, p_value = ttest_rel(df["ann"], df["linreg"])
    z = df["ann"] -  df["linreg"]
    t_stat = np.mean(z) / (np.std(z) / (len(z)**(1/2)))
    newZ = bootstrap10(z)
    lower_bound = np.percentile(newZ, 2.5)
    upper_bound = np.percentile(newZ, 97.5)
    print(lower_bound,upper_bound)
    print(f"t = {t_stat:.3f}, p = {p_value}")

    # Set a random seed for reproducibility
    np.random.seed(42)

    return t_stat,p_value,lower_bound,upper_bound


if __name__=="__main__":
    main()
