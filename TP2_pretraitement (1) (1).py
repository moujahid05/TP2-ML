"""TP2 — Prétraitement avancé d'un jeu de données en Machine Learning (IAA M1).

Exécution : python TP2_pretraitement.py
Dépendances : pandas, numpy, matplotlib, scikit-learn, joblib.
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

CSV = "/mnt/user-data/uploads/transactions_TP2.csv"
OUT = "/tmp/claude-0/-home-claude/58f73cf1-af87-5d1f-be90-e4333c9c7ea1/scratchpad/tp2"
os.makedirs(OUT, exist_ok=True)

RS = 42
NUM = ["montant_dh", "anciennete_mois", "nb_ops_24h", "heure"]
CAT = ["canal", "appareil", "region", "international"]
BILAN = {}


def titre(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def build_preproc(scaler):
    """Pipeline numérique (médiane + mise à l'échelle) et catégoriel (modale + one-hot)."""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)])
    cat = Pipeline(
        [
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


# Rééchantillonnage : équivalents de imbalanced-learn, implémentés avec numpy
def sur_echantillonnage(X, y, rng):
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    extra = rng.choice(c1, size=len(c0) - len(c1), replace=True)
    idx = np.concatenate([c0, c1, extra])
    return X[idx], y[idx], extra


def sous_echantillonnage(X, y, rng):
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    keep0 = rng.choice(c0, size=len(c1), replace=False)
    idx = np.concatenate([keep0, c1])
    return X[idx], y[idx], len(c0) - len(c1)


def smote(X_min, n_new, k, rng):
    """SMOTE : point synthétique = x + lambda * (voisin - x), lambda dans [0, 1]."""
    d = np.sqrt(((X_min[:, None, :] - X_min[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    voisins = np.argsort(d, axis=1)[:, :k]
    base = rng.integers(0, len(X_min), n_new)
    voisin = voisins[base, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1))
    return X_min[base] + lam * (X_min[voisin] - X_min[base])


# =====================================================================
# PARTIE 1 — Explorer et vérifier
# =====================================================================
titre("PARTIE 1 — Explorer et vérifier les données")

df = pd.read_csv(CSV, encoding="utf-8-sig")
print("1. Forme du fichier :", df.shape)
print(df.dtypes.to_string())
print(df.head().to_string())

titre("2. Valeurs manquantes par colonne (trié du plus grand au plus petit)")
na = pd.DataFrame(
    {"manquants": df.isna().sum(), "pourcentage": (df.isna().mean() * 100).round(2)}
)
print(na.sort_values("pourcentage", ascending=False).to_string())
BILAN["valeurs_manquantes_total_brut"] = int(df.isna().sum().sum())

titre("3. Doublons sur id_operation")
dup = df.duplicated(subset="id_operation", keep="first")
print("Doublons sur id_operation :", int(dup.sum()))
print("Doublons de lignes complètes :", int(df.duplicated().sum()))
n0 = len(df)
df = df[~dup].reset_index(drop=True)
print(f"Lignes : {n0} -> {len(df)}")
BILAN["doublons_supprimes"] = int(dup.sum())
BILAN["lignes_apres_dedoublonnage"] = len(df)

titre("4. Montant : statistiques, histogramme et boxplot")
m_all = df["montant_dh"].dropna()
print(m_all.describe().round(2).to_string())
print("Plus grands montants :", m_all.nlargest(5).round(2).tolist())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(m_all, bins=80, color="#4C72B0")
ax[0].set_title("Histogramme de montant_dh")
ax[0].set_xlabel("DH")
ax[1].boxplot(m_all)
ax[1].set_title("Boxplot de montant_dh")
ax[1].set_ylabel("DH")
fig.tight_layout()
fig.savefig(f"{OUT}/01_montant_histogramme_boxplot.png", dpi=120)
plt.close(fig)

titre("5. Déséquilibre des classes")
cnt = df["fraude"].value_counts().sort_index()
print(cnt.to_string())
print(f"Proportion de fraudes : {cnt[1] / cnt.sum() * 100:.2f} %")
print(f"Ratio normal / fraude : {cnt[0] / cnt[1]:.2f} : 1")
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["normale (0)", "fraude (1)"], cnt.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(cnt.values):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set_title("Répartition des deux classes")
fig.tight_layout()
fig.savefig(f"{OUT}/02_classes.png", dpi=120)
plt.close(fig)
BILAN["ratio_normal_sur_fraude"] = round(cnt[0] / cnt[1], 2)
BILAN["taux_fraude_pct"] = round(cnt[1] / cnt.sum() * 100, 2)

titre("6. Lien entre resultat_enquete et fraude")
print(pd.crosstab(df["resultat_enquete"], df["fraude"]).to_string())

# =====================================================================
# PARTIE 2 — Préparer les données sans fuite
# =====================================================================
titre("PARTIE 2 — Préparer les données sans fuite")

X = df.drop(columns=["id_operation", "id_client", "resultat_enquete", "fraude"])
y = df["fraude"]
print("7. X :", X.shape, "| colonnes :", list(X.columns))

X_ref, X_ver, y_ref, y_ver = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RS
)
print("8. Groupe de référence :", X_ref.shape, "| groupe de vérification :", X_ver.shape)
for nom, yy in [("référence", y_ref), ("vérification", y_ver)]:
    print(f"   fraude dans le {nom} : {yy.sum()} / {len(yy)} = {yy.mean() * 100:.2f} %")

titre("9. Types de variables")
print("Numériques  :", NUM)
print("Catégorielles :", CAT)

titre("10-12. Pipelines et ColumnTransformer")
pre_std = build_preproc(StandardScaler())
pre_std.fit(X_ref)  # médianes, modes, moyennes et écarts-types : groupe de référence seulement
A_ref = pre_std.transform(X_ref)
A_ver = pre_std.transform(X_ver)
print("Colonnes brutes en entrée :", X_ref.shape[1])
print("Colonnes en sortie :", A_ref.shape[1])
print("Lignes -> référence", A_ref.shape[0], "| vérification", A_ver.shape[0])
print("Noms des colonnes :", list(pre_std.get_feature_names_out()))
print(
    "Valeurs manquantes après transformation (réf, vérif) :",
    int(np.isnan(A_ref).sum()),
    int(np.isnan(A_ver).sum()),
)

titre("13. StandardScaler vs RobustScaler sur montant_dh (paramètres de la référence)")
imp_m = SimpleImputer(strategy="median").fit(X_ref[["montant_dh"]])
m_ref_imp = imp_m.transform(X_ref[["montant_dh"]])
sc_std = StandardScaler().fit(m_ref_imp)
sc_rob = RobustScaler().fit(m_ref_imp)
print(
    f"Standard : moyenne={sc_std.mean_[0]:.2f}, écart-type={np.sqrt(sc_std.var_[0]):.2f}"
)
print(f"Robust   : médiane={sc_rob.center_[0]:.2f}, IQR={sc_rob.scale_[0]:.2f}")
probe = np.array([[X_ref["montant_dh"].median()], [X_ref["montant_dh"].max()]])
tab13 = pd.DataFrame(
    {
        "montant brut": probe.ravel(),
        "StandardScaler": sc_std.transform(probe).ravel(),
        "RobustScaler": sc_rob.transform(probe).ravel(),
    },
    index=["opération ordinaire (médiane)", "montant très élevé (max)"],
)
print(tab13.round(3).to_string())

# =====================================================================
# PARTIE 3 — Comparer les méthodes de mise à l'échelle
# =====================================================================
titre("PARTIE 3 — Mise à l'échelle et valeurs extrêmes")

m_ref = X_ref["montant_dh"].dropna()
q1, q3 = m_ref.quantile([0.25, 0.75])
iqr = q3 - q1
b_bas, b_haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"15. Q1 = {q1:.2f} | Q3 = {q3:.2f} | IQR = {iqr:.2f}")
print(f"16. Bornes de Tukey : [{b_bas:.2f} ; {b_haut:.2f}]")

hors = (X_ref["montant_dh"] < b_bas) | (X_ref["montant_dh"] > b_haut)
print(
    f"    Montants hors bornes (référence) : {int(hors.sum())} sur {len(m_ref)} "
    f"({hors.mean() * 100:.1f} %)"
)
print("    Exemples (les plus grands) :")
verif = X_ref[hors].assign(fraude=y_ref[hors]).sort_values("montant_dh", ascending=False)
print(
    verif[["montant_dh", "canal", "appareil", "region", "international", "nb_ops_24h", "fraude"]]
    .head(8)
    .to_string()
)
print(
    f"    Taux de fraude : {y_ref[hors].mean() * 100:.1f} % parmi les extrêmes, "
    f"{y_ref[~hors].mean() * 100:.1f} % parmi les autres"
)
BILAN["montants_extremes_tukey_reference"] = int(hors.sum())

titre("17. Conserver ou limiter les montants extrêmes")
cap = m_ref.quantile(0.99)
print(f"Seuil de limitation = 99e centile calculé sur la référence : {cap:.2f} DH")
print(f"Montants limités (référence) : {int((X_ref['montant_dh'] > cap).sum())}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].boxplot(m_ref)
ax[0].set_title("Montants bruts (référence)")
ax[1].boxplot(m_ref.clip(upper=cap))
ax[1].set_title(f"Montants limités à {cap:.0f} DH")
fig.tight_layout()
fig.savefig(f"{OUT}/03_boxplots_brut_vs_limite.png", dpi=120)
plt.close(fig)

titre("18. MinMax, Standard et Robust : ajustés sur la référence, appliqués à la vérification")
ref_2d = m_ref.to_frame()
ver_2d = X_ver["montant_dh"].dropna().to_frame()
mm = MinMaxScaler().fit(ref_2d)
ss = StandardScaler().fit(ref_2d)
rb = RobustScaler().fit(ref_2d)
scalers = {"MinMax": mm, "Standard": ss, "Robust": rb}
print(f"MinMax   : min={mm.data_min_[0]:.2f}, max={mm.data_max_[0]:.2f}")
print(f"Standard : moyenne={ss.mean_[0]:.2f}, écart-type={np.sqrt(ss.var_[0]):.2f}")
print(f"Robust   : médiane={rb.center_[0]:.2f}, IQR={rb.scale_[0]:.2f}")
print("Vérification transformée (exemple) :", {k: sc.transform(ver_2d[:3]).ravel().round(3).tolist() for k, sc in scalers.items()})

titre("19. Montants habituels vs montant très élevé")
valeurs = pd.DataFrame({"montant_dh": [300.0, 700.0, 1200.0, float(m_ref.max())]})
t19 = pd.DataFrame(
    {"brut": valeurs["montant_dh"].to_numpy()}
    | {k: sc.transform(valeurs).ravel() for k, sc in scalers.items()},
    index=["habituel 300 DH", "habituel 700 DH", "habituel 1200 DH", "très élevé (max)"],
)
print(t19.round(3).to_string())
print("\nÉtendue interquartile des valeurs transformées (référence) :")
for k, sc in scalers.items():
    t = sc.transform(ref_2d).ravel()
    iq = np.subtract(*np.percentile(t, [75, 25]))
    print(f"  {k:8s} IQR = {iq:.4f} | min = {t.min():.3f} | max = {t.max():.3f}")

titre("20. Histogrammes du montant transformé")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, (k, sc) in zip(ax, scalers.items()):
    a.hist(sc.transform(ref_2d).ravel(), bins=80, color="#4C72B0")
    a.set_title(k)
fig.tight_layout()
fig.savefig(f"{OUT}/04_histogrammes_transformes.png", dpi=120)
plt.close(fig)

titre("21. Une valeur supérieure au max de la référence avec MinMax")
ex = MinMaxScaler().fit([[100.0], [300.0]])
print("Exemple : min = 100, max = 300 ; nouvelle valeur 450 ->", ex.transform([[450.0]])[0, 0])
print(
    "Montant réel plus grand que le max de la référence (40 000 DH) ->",
    round(float(mm.transform(pd.DataFrame({"montant_dh": [40000.0]}))[0, 0]), 3),
)

# =====================================================================
# PARTIE 4 — Classes déséquilibrées
# =====================================================================
titre("PARTIE 4 — Classes déséquilibrées (sur la référence uniquement)")

rng = np.random.default_rng(RS)
Xr = A_ref  # référence préparée (StandardScaler + one-hot)
yr = y_ref.to_numpy()
classes, cnts = np.unique(yr, return_counts=True)
print("22. Référence :", dict(zip(classes.tolist(), cnts.tolist())))
print(f"    Rapport normal / fraude = {cnts[0] / cnts[1]:.2f} : 1")

titre("23. RandomOverSampler (suréchantillonnage)")
Xo, yo, extra = sur_echantillonnage(Xr, yr, rng)
n0_, n1_ = cnts
print(f"Avant : normal={n0_}, fraude={n1_}")
print(f"Après : normal={(yo == 0).sum()}, fraude={(yo == 1).sum()}")
nouvelles = Xo[len(yr):]
print("Nouvelles lignes = copies exactes de lignes existantes :", np.array_equal(nouvelles, Xr[extra]))

titre("24. RandomUnderSampler (sous-échantillonnage)")
Xu, yu, supprimees = sous_echantillonnage(Xr, yr, rng)
print(f"Lignes conservées : normal={(yu == 0).sum()}, fraude={(yu == 1).sum()}")
print(f"Lignes supprimées (normal) : {supprimees}")

titre("25. SMOTE : exemple en 2 dimensions (montant et ancienneté standardisés)")
i_min = np.where(yr == 1)[0]
a, b = Xr[i_min[0], :2], Xr[i_min[1], :2]
lam = 0.4
s = a + lam * (b - a)
print(f"Observation réelle A : {a.round(3)}")
print(f"Observation réelle B : {b.round(3)}")
print(f"Synthétique (lambda = {lam}) : {s.round(3)}")
print("Le point synthétique est sur le segment [A, B], ce n'est pas une copie.")

titre("26. SMOTE appliqué au one-hot : valeurs non binaires")
Xmin = Xr[yr == 1]
n_new = cnts[0] - cnts[1]
syn = smote(Xmin, n_new, k=5, rng=rng)
idx_onehot = list(range(len(NUM), Xr.shape[1]))
mask_nb = ~np.isin(syn[:, idx_onehot], [0.0, 1.0])
print(f"Points synthétiques générés : {n_new}")
print(
    f"Points avec au moins une colonne one-hot fractionnaire (ex. 0,37 au lieu de 0 ou 1) : "
    f"{mask_nb.any(axis=1).mean() * 100:.1f} %"
)
print("Solution : SMOTE-NC (nombres + catégories), ou SMOTE sur les nombres seulement.")

titre("27. Effectifs avant / après rééquilibrage")
tab27 = pd.DataFrame(
    {
        "normal (0)": [cnts[0], (yo == 0).sum(), (yu == 0).sum(), cnts[0]],
        "fraude (1)": [cnts[1], (yo == 1).sum(), (yu == 1).sum(), cnts[0]],
    },
    index=["avant", "après suréchantillonnage", "après sous-échantillonnage", "après SMOTE (démo)"],
)
print(tab27.to_string())
print("Groupe de vérification (ne doit PAS changer) :", dict(zip(*np.unique(y_ver, return_counts=True))))

# =====================================================================
# PARTIE 5 — Version finale, vérification et sauvegarde
# =====================================================================
titre("PARTIE 5 — Version finale")

final = build_preproc(RobustScaler())
final.fit(X_ref)  # paramètres calculés sur la référence uniquement
F_ref = final.transform(X_ref)
F_ver = final.transform(X_ver)
names = final.get_feature_names_out()
print("28-29. Transformateur final : médiane + RobustScaler ; modale + one-hot")
print("Colonnes en sortie :", F_ref.shape[1])
print("Lignes -> référence", F_ref.shape[0], "| vérification", F_ver.shape[0])
print("Valeurs manquantes (réf, vérif) :", int(np.isnan(F_ref).sum()), int(np.isnan(F_ver).sum()))

titre("30. Catégorie inconnue de canal")
unk = X_ver.iloc[[0]].copy()
unk["canal"] = "chat_vocal"
row = final.transform(unk)[0]
canal_cols = [(n, round(v, 3)) for n, v in zip(names, row) if "canal" in n]
print("Canal inconnu transformé sans erreur. Colonnes canal :", canal_cols)

titre("31. Sauvegarde et rechargement avec joblib")
path = f"{OUT}/preprocesseur_final.joblib"
joblib.dump(final, path)
reloaded = joblib.load(path)
same = np.allclose(reloaded.transform(X_ver.iloc[:5]), F_ver[:5])
print("Fichier :", path)
print("Mêmes valeurs après rechargement :", bool(same))

pd.DataFrame(F_ref, columns=names).assign(fraude=y_ref.values).to_csv(
    f"{OUT}/reference_preparee.csv", index=False
)
pd.DataFrame(F_ver, columns=names).assign(fraude=y_ver.values).to_csv(
    f"{OUT}/verification_preparee.csv", index=False
)

titre("Bonus A — moyenne vs médiane pour l'imputation de montant_dh")
s_ref = X_ref["montant_dh"]
s_ext = s_ref.where(s_ref != s_ref.max(), 100000.0)
print(f"Moyenne normale : {s_ref.mean():.2f} | médiane : {s_ref.median():.2f}")
print(f"Si le max devient 100 000 DH : moyenne = {s_ext.mean():.2f} | médiane = {s_ext.median():.2f}")
print("La médiane ne bouge quasiment pas : elle résiste mieux aux montants très élevés.")

titre("BILAN")
BILAN["lignes_finales_reference"] = int(F_ref.shape[0])
BILAN["lignes_finales_verification"] = int(F_ver.shape[0])
BILAN["colonnes_finales"] = int(F_ref.shape[1])
print(json.dumps(BILAN, indent=2, ensure_ascii=False))
with open(f"{OUT}/bilan.json", "w", encoding="utf-8") as f:
    json.dump(BILAN, f, indent=2, ensure_ascii=False)
print("\nFichiers produits dans :", OUT)
