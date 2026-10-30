import s3fs
import pandas as pd
import numpy as np

# Preferred: create the shared AWS profile once, then reuse it here.
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)

obr = pd.read_parquet("s3://lfaivre/diffusion/aus1league_obraggregates_20242025.parquet", filesystem=fs)
passing = pd.read_parquet("s3://lfaivre/diffusion/aus1league_passingaggregates_20242025.parquet", filesystem=fs)
physical = pd.read_parquet("s3://lfaivre/diffusion/aus1league_physicalaggregates_20242025.parquet", filesystem=fs)

# ------------------------------------------------------------------ PARAMÈTRES
MIN_MATCHES = 3                              # on ignore les joueurs ayant moins de N matchs
ANNEE_REF = 2025 + 5 / 12                    # ≈ 31 mai 2025 (fin de saison), pour calculer l'âge

# ------------------------------------------------------------------ 1. FILTRE ET COPIES
# Filtre par condition logique : [INTRO §6.3 Filter] df.loc[condition] ; [STATS §2.2] communes = cog_2023.loc[cog_2023['TYPECOM']=="COM"]
# .copy() : [STATS §4.2 Exercice 2] df_wide = emissions_wide.copy()
df_phys = physical.loc[physical["count_match"] >= MIN_MATCHES].copy()
df_pas = passing.loc[passing["performance_count"] >= MIN_MATCHES].copy()
df_obr = obr.loc[obr["performance_count"] >= MIN_MATCHES].copy()

# ------------------------------------------------------------------ 2. NOMS D'ÉQUIPES RACCOURCIS
# Création d'une colonne par réassignation : [INTRO §6.1] df['dep'] = df['INSEE commune'].str[:2]
# .map(lambda ...) : [STATS §5 Formatter des tableaux] emissions_table[...].map(lambda x: create_bar(...))
# .replace / .strip : méthodes de chaînes de Python de base [PY]
for df in [df_phys, df_pas, df_obr]:         # boucle for [PY]
    df["equipe"] = df["team_name"].map(lambda s: s.replace(" Football Club", "").replace(" FC", "").strip())

# ------------------------------------------------------------------ 3. POIDS (volume de jeu réel du joueur)
# Opérations entre colonnes + réassignation : [INTRO §6.1] ; produit de deux colonnes comme dans [STATS §5] emissions / POPULATION
df_phys["w_minutes"] = df_phys["minutes_full_all"] * df_phys["count_match"]          # physique : minutes jouées au total
for df in [df_pas, df_obr]:
    df["w_possession"] = df["minutes_tip"] * df["performance_count"]                  # passes / courses : minutes de possession
df_pas["w_passes"] = df_pas["pass_count_attempted_total"]                             # % de passes : pondéré par les passes tentées
df_obr["w_courses"] = df_obr["offballrun_count_total"]                                # distance des courses : pondérée par le nombre de courses

# Âge approximatif (date de référence fixée et arrondie au mois) : on découpe la date "AAAA-MM-JJ" avec .str[:4] puis .astype(float)
# .str[:2] : [INTRO §6.1] ; .astype(...) : [INTRO §2.2 Avant le DataFrame, la Serie] et [STATS §1.1] .astype(int)
annee = df_phys["player_birthdate"].str[:4].astype(float)
mois = df_phys["player_birthdate"].str[5:7].astype(float)
df_phys["age"] = ANNEE_REF - (annee + (mois - 1) / 12)


# ------------------------------------------------------------------ 4. OUTILS (fonctions)
# def : [STATS §5] def create_bar(...) ; f-strings : [STATS §1.1] f"https://api.insee.fr/melodi/data/{dataset}"
def moyennes_ponderees(df, variables, poids, bloc):
    """Moyenne pondérée par équipe = somme(valeur x poids) / somme(poids), en ignorant les valeurs manquantes."""
    tmp = df.copy()                                           # [STATS §4.2]
    agregats = {}                                             # dictionnaire [PY]
    for var in variables:
        present = 1 - tmp[var].isna().astype(int)             # [INTRO §7.3] isna ; [INTRO §2.2] astype : convertit les booléens en 0/1
        tmp[var + "_num"] = tmp[var].fillna(0) * tmp[poids] * present   # [INTRO §7.3] fillna : valeurs absentes remplacées par zéro dans le numérateur
        tmp[var + "_den"] = tmp[poids] * present
        agregats[var + "_num"] = "sum"
        agregats[var + "_den"] = "sum"
    # groupby + agg avec un dictionnaire : [STATS §2.3] filosofi.groupby('dep').agg({"POPULATION": "sum"})
    somme = tmp.groupby("equipe").agg(agregats)
    colonnes = []
    for var in variables:
        nom = f"{bloc}__{var}__moyenne_pond"
        somme[nom] = somme[var + "_num"] / somme[var + "_den"]          # division de colonnes : [STATS §5]
        colonnes.append(nom)
    return somme.loc[:, colonnes]                             # sélection de colonnes : [INTRO §5.2] df.loc[:, ['variable1','variable2']]


def stat_par_equipe(df, variables, stat, bloc):
    """Une statistique ('median', 'std' ou 'max') par équipe pour une liste de variables."""
    dico = {}
    for var in variables:
        dico[var] = stat
    # agg({variable: "median"}) : [STATS §5] .groupby("dep").agg({"empreinte": "sum", "revenu": "median", "population": "sum"})
    # "std" : [INTRO §7.2] df.agg({'Résidentiel': ['mean', 'std'], ...}) ; max : [STATS §5] emissions_table[variable].max()
    sortie = df.groupby("equipe").agg(dico)
    noms_stats = {"median": "mediane", "std": "ecart_type", "max": "max"}
    renommage = {}
    for var in variables:
        renommage[var] = f"{bloc}__{var}__{noms_stats[stat]}"
    # rename avec un dictionnaire : [STATS §5] .rename(columns={...}) et [INTRO §6.1] rename({...}, axis=1)
    return sortie.rename(columns=renommage)


def fusionner(liste_tableaux):
    """Assemble des tableaux indexés par équipe en un seul, avec une ligne par équipe."""
    # reset_index : [STATS §2.2] ; merge(..., on=..., how="outer") : [JOINTURES §4.4 Full join] et [STATS §1.1 Données]
    resultat = liste_tableaux[0].reset_index()
    for tableau in liste_tableaux[1:]:
        resultat = resultat.merge(tableau.reset_index(), on="equipe", how="outer")
    return resultat


# ------------------------------------------------------------------ 5. CONTEXTE
df_phys["un"] = 1                                              # colonne de 1 pour compter les lignes (réassignation, [INTRO §6.1])
# agg avec "nunique" : [STATS §2.2] communes.groupby('DEP').agg({'COM': 'nunique'}) ; "sum" : [STATS §2.3]
contexte = df_phys.groupby("equipe").agg({"player_id": "nunique", "un": "sum", "w_minutes": "sum"})
contexte = contexte.rename(columns={"player_id": "contexte__n_joueurs",
                                    "un": "contexte__n_profils_joueur_poste",
                                    "w_minutes": "contexte__minutes_totales"})
age = moyennes_ponderees(df_phys, ["age"], "w_minutes", "contexte")
age = age.rename(columns={"contexte__age__moyenne_pond": "contexte__age_moyen_pond"})

# Part des minutes par poste : une colonne de minutes par poste, puis division par le total
postes = ["Center Forward", "Central Defender", "Full Back", "Midfield", "Wide Attacker"]
# contrôle : df_phys["position_group"].nunique() doit valoir len(postes)  ([INTRO §5.1] nunique)
dico_postes = {"w_minutes": "sum"}
for poste in postes:
    # comparaison -> booléens : [INTRO §3.1 L'indexation] animaux['animal'] == "chat" ; booléen x nombre = 0 ou le nombre
    df_phys["min_" + poste] = df_phys["w_minutes"] * (df_phys["position_group"] == poste)
    dico_postes["min_" + poste] = "sum"
tab_postes = df_phys.groupby("equipe").agg(dico_postes)
colonnes_postes = []
for poste in postes:
    nom = f"contexte__part_minutes__{poste}"
    tab_postes[nom] = tab_postes["min_" + poste] / tab_postes["w_minutes"]
    colonnes_postes.append(nom)
tab_postes = tab_postes.loc[:, colonnes_postes]

# ------------------------------------------------------------------ 6. PHYSIQUE
vars_phys = ["total_distance_full_all", "total_metersperminute_full_all",
             "hsr_distance_full_all", "sprint_distance_full_all", "sprint_count_full_all",
             "highaccel_count_full_all", "highdecel_count_full_all",
             "total_metersperminute_full_tip", "total_metersperminute_full_otip",
             "hsr_distance_full_tip", "hsr_distance_full_otip", "psv99"]
phys_moy = moyennes_ponderees(df_phys, vars_phys, "w_minutes", "physique")
phys_med = stat_par_equipe(df_phys, vars_phys, "median", "physique")
phys_std = stat_par_equipe(df_phys, vars_phys, "std", "physique")
phys_max = stat_par_equipe(df_phys, ["psv99"], "max", "physique")     # vitesse de pointe : le maximum est parlant

# ------------------------------------------------------------------ 7. PASSES
vars_pas = ["pass_count_attempted_p30tip", "pass_count_completed_p30tip",
            "pass_count_longrange_attempted_p30tip", "pass_count_onetouch_attempted_p30tip",
            "pass_count_linebreak_attempted_p30tip", "pass_count_linebreak_completed_p30tip",
            "pass_count_dangerous_attempted_p30tip", "pass_count_dangerous_completed_p30tip",
            "pass_count_difficultpass_attempted_p30tip", "pass_count_shotwithin10s_p30tip"]
vars_pas_pct = ["pass_pct_completed", "pass_avgdistance"]
pas_moy1 = moyennes_ponderees(df_pas, vars_pas, "w_possession", "passes")
pas_moy2 = moyennes_ponderees(df_pas, vars_pas_pct, "w_passes", "passes")
pas_med = stat_par_equipe(df_pas, vars_pas + vars_pas_pct, "median", "passes")
pas_std = stat_par_equipe(df_pas, vars_pas + vars_pas_pct, "std", "passes")

# ------------------------------------------------------------------ 8. COURSES SANS BALLON
vars_obr = ["offballrun_count_p30tip", "offballrun_count_targeted_p30tip",
            "offballrun_count_received_p30tip", "offballrun_count_dangerous_p30tip",
            "offballrun_count_abovehsr_p30tip", "offballrun_count_penaltyarea_p30tip",
            "offballrun_count_shotwithin10s_p30tip"]
vars_obr_dist = ["offballrun_avgdistance"]
obr_moy1 = moyennes_ponderees(df_obr, vars_obr, "w_possession", "courses_sans_ballon")
obr_moy2 = moyennes_ponderees(df_obr, vars_obr_dist, "w_courses", "courses_sans_ballon")
obr_med = stat_par_equipe(df_obr, vars_obr + vars_obr_dist, "median", "courses_sans_ballon")
obr_std = stat_par_equipe(df_obr, vars_obr + vars_obr_dist, "std", "courses_sans_ballon")

# Types de courses : moyennes pondérées, puis part de chaque type dans l'ensemble des courses
types = ["behind", "comingshort", "crossreceiver", "droppingoff", "overlap",
         "pullinghalfspace", "pullingwide", "aheadoftheball", "support", "underlap"]
vars_types = [t + "run_count_p30tip" for t in types]           # liste construite par compréhension : [INTRO §6.1] [v + "_log" for v in vars]
types_moy = moyennes_ponderees(df_obr, vars_types, "w_possession", "types_courses")

# ------------------------------------------------------------------ 9. ASSEMBLAGE
stats_equipes = fusionner([contexte, age, tab_postes,
                           phys_moy, phys_med, phys_std, phys_max,
                           pas_moy1, pas_moy2, pas_med, pas_std,
                           obr_moy1, obr_moy2, obr_med, obr_std,
                           types_moy]).copy()          # .copy() : [STATS §4.2] ; évite un avertissement de pandas (trop de colonnes ajoutées une à une)


def col(bloc, var):
    return f"{bloc}__{var}__moyenne_pond"


# Part de chaque type de course : [STATS §5] raw_perc = variable / variable.max() ; [INTRO §7.2] 100*x/x.sum() (même logique de part)
total_courses = stats_equipes[col("courses_sans_ballon", "offballrun_count_p30tip")]
colonnes_a_retirer = []
for t in types:
    brut = col("types_courses", t + "run_count_p30tip")
    stats_equipes[f"types_courses__{t}__part"] = stats_equipes[brut] / total_courses
    colonnes_a_retirer.append(brut)
stats_equipes = stats_equipes.drop(columns=colonnes_a_retirer)    # drop(columns=...) : [INTRO §6.1]

# ------------------------------------------------------------------ 10. RATIOS D'ÉQUIPE (calculés sur les moyennes pondérées)
# Divisions et sommes de colonnes : [STATS §5] emissions_merged['emissions']/emissions_merged['POPULATION']
e = stats_equipes
e["ratios__intensite_otip_sur_tip"] = (e[col("physique", "total_metersperminute_full_otip")]
                                       / e[col("physique", "total_metersperminute_full_tip")])
e["ratios__part_hsr_hors_possession"] = (e[col("physique", "hsr_distance_full_otip")]
                                         / (e[col("physique", "hsr_distance_full_otip")] + e[col("physique", "hsr_distance_full_tip")]))
e["ratios__part_passes_longues"] = (e[col("passes", "pass_count_longrange_attempted_p30tip")]
                                    / e[col("passes", "pass_count_attempted_p30tip")])
e["ratios__part_passes_lignes_cassees"] = (e[col("passes", "pass_count_linebreak_attempted_p30tip")]
                                           / e[col("passes", "pass_count_attempted_p30tip")])
e["ratios__part_passes_dangereuses"] = (e[col("passes", "pass_count_dangerous_attempted_p30tip")]
                                        / e[col("passes", "pass_count_attempted_p30tip")])
e["ratios__reussite_lignes_cassees"] = (e[col("passes", "pass_count_linebreak_completed_p30tip")]
                                        / e[col("passes", "pass_count_linebreak_attempted_p30tip")])
e["ratios__taux_courses_servies"] = (e[col("courses_sans_ballon", "offballrun_count_targeted_p30tip")]
                                     / e[col("courses_sans_ballon", "offballrun_count_p30tip")])
e["ratios__taux_reception_sur_ciblees"] = (e[col("courses_sans_ballon", "offballrun_count_received_p30tip")]
                                           / e[col("courses_sans_ballon", "offballrun_count_targeted_p30tip")])
e["ratios__part_courses_dangereuses"] = (e[col("courses_sans_ballon", "offballrun_count_dangerous_p30tip")]
                                         / e[col("courses_sans_ballon", "offballrun_count_p30tip")])
e["ratios__part_courses_haute_vitesse"] = (e[col("courses_sans_ballon", "offballrun_count_abovehsr_p30tip")]
                                           / e[col("courses_sans_ballon", "offballrun_count_p30tip")])

# ------------------------------------------------------------------ 11. TRI, AFFICHAGE ET EXPORT
# sort_values : [INTRO §6.2] ; reset_index(drop=True) : [STATS §5] ; shape : [INTRO §5.1] ; head : [INTRO §6.1]
stats_equipes = stats_equipes.sort_values("equipe").reset_index(drop=True)
print(stats_equipes.shape)
print(stats_equipes.head(3))
# .round() : [INTRO §7.2] ; ex. : stats_equipes.round(3)

stats_equipes.to_csv("stats_equipes.csv", sep=";", decimal=",", encoding="utf-8-sig", index=False)