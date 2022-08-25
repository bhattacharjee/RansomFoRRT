import matplotlib
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from functools import lru_cache
import gc
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import glob
import tqdm
from multiprocessing.pool import ThreadPool, Pool

def get_weighted_importances_1(all_results):
    f1_colname = None
    importance_colname = None
    for k in all_results[0]:
        if "f1_score" in k:
            f1_colname = k
        if "feature_importance" in k:
            importance_colname = k
    f1_scores = []
    importances = []
    print(f1_colname, importance_colname)
    for result in all_results:
        f1_scores.append(result[f1_colname])
        importances.append(result[importance_colname])
    
    out_importances = np.zeros(len(importances[0]))
    for f1, imp in zip(f1_scores, importances):
        out_importances = out_importances + imp * f1
    out_importances = out_importances / sum(f1_scores)
    
    return out_importances

with open("simple-average.pickle", "rb") as f:
	columns, all_results = pickle.load(f)

plt.rcParams["figure.figsize"] = (10,20)
plt.rcParams['font.size'] = 12
plt.rc('font', size=42) 
df2_dict = {"features": columns, "importances": get_weighted_importances_1(all_results)}
imp_df = pd.DataFrame(df2_dict)
imp_df = imp_df.sort_values(by="importances", ascending=False).reset_index(drop="True")
sns.set_theme(style="white")
def select_moments(x):
    if "baseline" in x:
        return "Baseline Features"
    elif "advanced" in x:
        return "Advanced Statistics"
    elif "fourier" in x:
        return "Fourier Spectrum"
    else:
        return "unknown"
sns.color_palette("rocket", as_cmap=True)
imp_df["Type of Feature"] = imp_df["features"].map(select_moments)
imp_df = imp_df[imp_df["importances"] > 0.0005]
imp_df["Feature"] = imp_df["features"]
imp_df["Importance"] = imp_df["importances"]
sns.barplot(y="Feature", x="Importance", data=imp_df, hue="Type of Feature", dodge=False, palette="bright")
plt.show()
