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

plt.rcParams["figure.figsize"] = (15,10)
#plt.rcParams['font.size'] = 8
#plt.subplots_adjust(left=10)
plt.subplots_adjust(left=0.186, bottom=0.107, right=0.552, top=0.883, wspace=0.2, hspace=0.2)
#plt.rc('font', size=14) 
df2_dict = {"features": columns, "importances": get_weighted_importances_1(all_results)}
imp_df = pd.DataFrame(df2_dict)
imp_df = imp_df.sort_values(by="importances", ascending=False).reset_index(drop="True")
sns.set_theme(style="white")
def select_moments(x):
    if "baseline" in x or ("dit" in x and "shanon" in x):
        return "Baseline Features"
    elif "advanced" in x:
        return "Advanced Statistics"
    elif "fourier" in x:
        return "Fourier Spectrum"
    else:
        return "unknown"

rename_map = {
    "fourier.stat.1byte.autocorr": "Fourier autocorrelation (1 byte)",
    "fourier.stat.1byte.std": "Fouerier STD (1 byte)",
    "advanced.skew_full": "Skewness",
    "fourier.stat.1byte.mean": "Fourier mean (1 byte)",
    "advanced.dit.renyi.full.inf": r"Rényi's entropy $(\alpha = \infty)$",
    "advanced.moment_full.4": r"$4^{th}$ moment",
    "advanced.moment_full.2": r"$2^{nd}$ moment",
    "advanced.dit.tsallis.full.1": r"Tsallis' entropy $(n = 1)$",
    "advanced.dit.tsallis.full.0": r"Tsallis' entropy $(n = 0)$",
    "advanced.moment_full.6": r"$6^{th}$ moment",
    "advanced.dit.shanon.full": "Shannon's entropy",
    "advanced.dit.renyi.full.0": r"Rényi's entropy $(\alpha = 0)$",
    "advanced.dit.renyi.full.9": r"Rényi's entropy $(\alpha = 9)$",
    "advanced.moment_full.8": r"$8^{th}$ moment",
    "baseline.montecarlo_pi": "Monte-Carlo simulation of $\pi$",
#"advanced.dit.renyi.full.1": r"Rényi's entropy $(\alpha = 1)$",
    "advanced.dit.renyi.full.1": r"delete",
    "baseline.shannon_entropy": "delete",
    "baseline.autocorrelation_full": "Autocorrelation",
    "advanced.dit.renyi.full.8": r"Rényi's entropy $(\alpha = 8)$",
    "advanced.kurtosis_full": "Kurtosis",
    "advanced.dit.renyi.full.10": r"Rényi's entropy $(\alpha = 10)$",
    "advanced.moment_full.12": r"$12^{th}$ moment",
    "fourier.stat.1byte.chisq": "Fourier Chi-Square statistic (1 byte)",
    "baseline.chisquare_full": "Chi-Square statistic",
    "advanced.dit.renyi.full.7": r"Rényi's entropy $(\alpha = 7)$",
    "advanced.dit.renyi.full.5": r"Rényi's entropy $(\alpha = 5)$",
    "fourier.stat.1byte.moment.2": r"Fourier $2^{nd}$ moment (1 byte)",
    "advanced.moment_full.10": r"$10^{th}$ moment",
    "advanced.dit.renyi.full.6": r"Rényi's entropy $(\alpha = 6)$",
    "advanced.dit.renyi.full.3": r"Rényi's entropy $(\alpha = 3)$",
    "advanced.moment_full.14": r"$14^{th}$ moment",
    "advanced.dit.renyi.full.2": r"Rényi's entropy $(\alpha = 2)$",
    "advanced.moment_full.3": r"$3^{rd}$ moment",
    "advanced.dit.extropy.full": "Extropy",
    "advanced.dit.renyi.full.4": r"Rényi's entropy $(\alpha = 4)$",
    "advanced.dit.tsallis.full.2": r"Tsallis' entropy $(n = 2)$",
    "advanced.moment_full.5": r"$5^{th}$ moment",
    "fourier.stat.1byte.moment.3": r"Fourier $3^{rd}$ moment (1 byte)",
    "advanced.dit.tsallis.full.3": r"Tsallis' entropy $(n = 3)$",
    "advanced.moment_full.7": r"$7^{th}$ moment",
    "advanced.moment_full.9": r"$9^{th}$ moment",
    "advanced.moment_full.11": r"$11^{th}$ moment",
    "advanced.moment_full.13": r"$13^{th}$ moment",
    "advanced.moment_full.15": r"$15^{th}$ moment",
    "advanced.dit.tsallis.full.4": r"Tsallis' entropy $(n = 4)$",
    "advanced.dit.tsallis.full.5": r"Tsallis' entropy $(n = 5)$",
    "fourier.stat.1byte.moment.4": r"Fourier $4^{th}$ moment (1 byte)",
    "advanced.dit.tsallis.full.6": r"Tsallis' entropy $(n = 6)$",
    "fourier.stat.1byte.moment.5": r"Fourier $5^{th}$ moment (1 byte)",
    "advanced.dit.tsallis.full.7": r"Tsallis' entropy $(n = 7)$",
    "advanced.dit.tsallis.full.8": r"Tsallis' entropy $(n = 8)$",
    "fourier.stat.1byte.moment.6": r"Fourier $6^{th}$ moment (1 byte)",
    "advanced.dit.tsallis.full.9": r"Tsallis' entropy $(n = 9)$",
    "fourier.stat.1byte.moment.7": r"Fourier $7^{th}$ moment (1 byte)",
    "advanced.dit.tsallis.full.10": r"Tsallis' entropy $(n = 10)$",
    "fourier.stat.1byte.moment.8": r"Fourier $8^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.9": r"Fourier $9^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.10": r"Fourier $10^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.11": r"Fourier $11^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.12": r"Fourier $12^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.13": r"Fourier $13^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.14": r"Fourier $14^{th}$ moment (1 byte)",
    "fourier.stat.1byte.moment.15": r"Fourier $15^{th}$ moment (1 byte)",
}

def rename_features(x):
    global rename_map
    if x not in rename_map.keys():
        return x
    if "delete" == rename_map[x]:
        return "delete"
    if "" == rename_map[x]:
        return x
    return rename_map[x]

sns.color_palette("rocket", as_cmap=True)
imp_df["Type of Feature"] = imp_df["features"].map(select_moments)
imp_df["features"] = imp_df["features"].map(rename_features)
imp_df = imp_df[imp_df["importances"] > 0.0005]
imp_df["Feature"] = imp_df["features"]
imp_df["Importance"] = imp_df["importances"]
imp_df = imp_df[imp_df["Feature"] != "delete"]
sns.barplot(y="Feature", x="Importance", data=imp_df, hue="Type of Feature", dodge=False, palette="bright")
plt.savefig('/Users/phantom/Downloads/feature_importances.png')
plt.show()
