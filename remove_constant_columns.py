import pandas as pd

def main():
    df = pd.read_excel("Data4_final.xlsx")
    to_drop = []
    for col in df.columns:
        if len(set(df[col])) == 1:
            to_drop.append(col)
    df = df.drop(columns=to_drop)
    df.to_excel("Data5_constant_columns_removed.xlsx",index=False)

if __name__=="__main__":
    main()
