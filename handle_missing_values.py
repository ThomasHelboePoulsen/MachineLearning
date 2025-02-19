import pandas as pd
import numpy as np

def main():
    df = pd.read_excel("Data3_encoded.xlsx")
    df.loc[np.any(df.isna(), axis=1)].to_excel("Data4_removedRows.xlsx",index=False)
    df.dropna().to_excel("Data4_final.xlsx",index=False)


if __name__ == "__main__":
    main()
