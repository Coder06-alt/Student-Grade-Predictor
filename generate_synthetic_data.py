"""Generate a SYNTHETIC student dataset with the same columns as the public
UCI 'Student Performance' dataset (student-mat.csv, 395 rows, target G3).

Why synthetic? It lets the project run anywhere. To use the REAL data:
download student-mat.csv from the UCI repository (dataset 320, "Student
Performance"), put it in data/ as student-mat.csv (semicolon separated is
handled by the notebook) and re-run the notebook. No code changes needed.

Run: python generate_synthetic_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 395


def pick(options, probs=None):
    return rng.choice(options, size=N, p=probs)


ability = rng.normal(0, 1, N)  # hidden "academic ability" driving grades

df = pd.DataFrame(
    {
        "school": pick(["GP", "MS"], [0.88, 0.12]),
        "sex": pick(["F", "M"]),
        "age": rng.integers(15, 22, N),
        "address": pick(["U", "R"], [0.78, 0.22]),
        "famsize": pick(["GT3", "LE3"], [0.71, 0.29]),
        "Pstatus": pick(["T", "A"], [0.9, 0.1]),
        "Medu": rng.integers(0, 5, N),
        "Fedu": rng.integers(0, 5, N),
        "Mjob": pick(["teacher", "health", "services", "at_home", "other"]),
        "Fjob": pick(["teacher", "health", "services", "at_home", "other"]),
        "reason": pick(["home", "reputation", "course", "other"]),
        "guardian": pick(["mother", "father", "other"], [0.7, 0.23, 0.07]),
        "traveltime": rng.integers(1, 5, N),
        "studytime": np.clip(np.round(2 + 0.5 * ability + rng.normal(0, 0.8, N)), 1, 4).astype(int),
        "failures": np.clip(np.round(0.6 - 0.6 * ability + rng.normal(0, 0.7, N)), 0, 3).astype(int),
        "schoolsup": pick(["yes", "no"], [0.13, 0.87]),
        "famsup": pick(["yes", "no"], [0.61, 0.39]),
        "paid": pick(["yes", "no"]),
        "activities": pick(["yes", "no"]),
        "nursery": pick(["yes", "no"], [0.8, 0.2]),
        "higher": pick(["yes", "no"], [0.95, 0.05]),
        "internet": pick(["yes", "no"], [0.83, 0.17]),
        "romantic": pick(["yes", "no"], [0.33, 0.67]),
        "famrel": rng.integers(1, 6, N),
        "freetime": rng.integers(1, 6, N),
        "goout": rng.integers(1, 6, N),
        "Dalc": np.clip(rng.poisson(0.6, N) + 1, 1, 5),
        "Walc": np.clip(rng.poisson(1.2, N) + 1, 1, 5),
        "health": rng.integers(1, 6, N),
        "absences": np.clip(np.round(rng.gamma(1.3, 4.5, N) - 0.8 * ability), 0, 75).astype(int),
    }
)

# Grades on a 0-20 scale: G1, G2 (period grades) strongly predict final grade G3
g1 = 10.5 + 3.0 * ability + 0.5 * (df.studytime - 2) - 0.8 * df.failures + rng.normal(0, 1.3, N)
g2 = g1 + 0.3 * ability + rng.normal(0, 1.2, N) - 0.02 * df.absences
g3 = 0.25 * g1 + 0.75 * g2 + rng.normal(0, 1.0, N) - 0.03 * df.absences
df["G1"] = np.clip(np.round(g1), 0, 20).astype(int)
df["G2"] = np.clip(np.round(g2), 0, 20).astype(int)
df["G3"] = np.clip(np.round(g3), 0, 20).astype(int)

# A few students drop out and get G3 = 0 (also happens in the real data)
dropouts = rng.choice(N, size=int(0.03 * N), replace=False)
df.loc[dropouts, "G3"] = 0

# Inject realistic missing values so the imputation step is genuinely needed
for col, frac in [("absences", 0.04), ("Medu", 0.03), ("Fedu", 0.03),
                  ("internet", 0.04), ("studytime", 0.02), ("Mjob", 0.03)]:
    idx = rng.choice(N, size=int(frac * N), replace=False)
    df[col] = df[col].astype(object) if df[col].dtype != object else df[col]
    df.loc[idx, col] = np.nan
for col in ["absences", "Medu", "Fedu", "studytime"]:
    df[col] = pd.to_numeric(df[col])

df.to_csv("data/student-mat.csv", index=False)
print(df.shape, "saved to data/student-mat.csv")
