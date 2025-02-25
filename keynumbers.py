# =============================================================================
# Format changer
# =============================================================================

# =============================================================================
# median, mean, std, max, min
# =============================================================================
import pandas as pd
import numpy as np


def main():
    data = pd.read_csv("Data4_final.csv")
    unencoded = pd.read_csv("Data2_ColumnFiltered.csv")
    
    X = data.to_numpy()
    
    # =============================================================================
    # Anvendelskode, årstal, areal
    # =============================================================================
    columnsData = ["byg026Opførelsesår", "byg038SamletBygningsareal", "byg404Koordinat_easting", "byg404Koordinat_northing", "virkningFra_unix"]
    columnsUnencoded = ["byg056Varmeinstallation"]
    
    medians = [data[p].median() for p in columnsData] + [unencoded[p].median() for p in columnsUnencoded]
    means =  [data[p].mean() for p in columnsData] + [unencoded[p].mean() for p in columnsUnencoded]
    stds =  [data[p].std() for p in columnsData] + [unencoded[p].std() for p in columnsUnencoded]
    maxes =  [max(data[p]) for p in columnsData] + [max(unencoded[p]) for p in columnsUnencoded]
    minimums =  [min(data[p]) for p in columnsData] + [min(unencoded[p]) for p in columnsUnencoded]
    ranges = [maxes[i] - minimums[i] for i in range(len(maxes))]
    
    collectedNumbers = pd.DataFrame([medians, means, stds, maxes, minimums, ranges], 
                                    columns=["Opførselsår","Samlet Bygningsareal", "Easting", "Northing", "Virkning Fra [Unix]", "Varmeinstallation"], 
                                    index = ["Median", "Mean", "STD", "Max", "Min", "Range"])
    
    collectedNumbers.to_excel("Summary Statistics.xlsx")

if __name__ == "__main__":
    main()