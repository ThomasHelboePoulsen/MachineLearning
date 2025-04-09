import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
from scipy import stats

def main():
    df = pd.read_excel("tTestResultsClassification.xlsx")
    #df = df.head(5000)
    comparisons = [["knn","logreg"],["knn","baseline"],["baseline","logreg"]]
    data = [(f"{c1} - {c2}", *differenceTest(df[c1],df[c2]) ) for c1,c2 in comparisons]
    df = pd.DataFrame(data,columns=["test","t_stat","p_value","difference 95% confidence interval"])
    print(df)
    df.to_excel("classifiaction_t_test.xlsx")



def differenceTest(vector1,vector2):
    """get difference measures using paired t test. Must be dependent ie. v1_i and v2_i must be errors when predicting on the same input"""
    t_stat, p_value = ttest_rel(vector1, vector2)
    z = vector1 -  vector2
    #t_stat = np.mean(z) / (np.std(z) / (len(z)**(1/2))) -- also works.
    mean_z = np.mean(z)
    sem_z = stats.sem(z)
    frihedsGrader = len(z) - 1
    critical_values_of_the_t_distribution = stats.t.ppf(0.975, frihedsGrader)
    margin_of_error = critical_values_of_the_t_distribution * sem_z
    return t_stat,p_value,[mean_z - margin_of_error,mean_z + margin_of_error]


if __name__=="__main__":
    main()
