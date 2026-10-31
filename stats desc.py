# %%
import s3fs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preferred: create the shared AWS profile once, then reuse it here.
fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)

obr = pd.read_parquet("s3://lfaivre/diffusion/aus1league_obraggregates_20242025.parquet", filesystem=fs)
passing = pd.read_parquet("s3://lfaivre/diffusion/aus1league_passingaggregates_20242025.parquet", filesystem=fs)
physical = pd.read_parquet("s3://lfaivre/diffusion/aus1league_physicalaggregates_20242025.parquet", filesystem=fs)

#on garde uniquement les joueurs à plus de 3 matchs (skill corner conseille de prendre 5 donc à voir)
MIN_MATCHES = 3

#on applique le filtre sur les 3 data frames et on en fait une copie 
phys = physical.loc[physical['count_match'] >= MIN_MATCHES].copy()
pas = passing.loc[passing['performance_count'] >= MIN_MATCHES].copy()
courses = obr.loc[obr['performance_count'] >= MIN_MATCHES].copy()

#on simplifie les noms des équipes pour alléger les sorties 
for df in [phys, pas, courses]:
    df['equipe'] = df['team_name'].map(
        lambda s: s.replace(' Football Club', '').replace(' FC', '').strip()
    )

#on va calculer des poids pour calculer ensuite des moyennes pondérées. On va donner de l'importance au temps de jeu : un joueur qui joue beaucoup est plus représentatif que celui qui joue très peu (lorsque l'on a des données par 30 minutes)
phys['poids'] = phys['minutes_full_all'] * phys['count_match']
pas['poids'] = pas['minutes_tip'] * pas['performance_count']
courses['poids'] = courses['minutes_tip'] * courses['performance_count']
pas['poids_passes'] = pas['pass_count_attempted_total']
courses['poids_courses'] = courses['offballrun_count_total']


#on définit une fonction pour calculer la moyenne pondérée pour nos différentes variables selon le data frame, les variables et les poids
#moyenne pondérée = somme(variable * poids)/somme(poids)
def moyenne_ponderee_par_equipe(df, variables, poids):
    resultat = {}

    for variable in variables:
        valeurs = df[variable].fillna(0) #on gere les valeurs manquantes, on met 0 au lieu de NA ce qui va nous permettre de calculer la somme
        poids_valides = df[poids].fillna(0) * (1 - df[variable].isna().astype(int)) #permet d'exclure les valeurs manquantes de la moyenne pondérée (donne poids*1 lorsque la valeur est là, 0 sinon)
        numerateur = (valeurs * poids_valides).groupby(df['equipe']).sum()
        denominateur = poids_valides.groupby(df['equipe']).sum()
        resultat[variable] = numerateur / denominateur
    return pd.DataFrame(resultat)

#fonction qui permet d'avoir la dispersion des profils de joueurs au sein des équipes
def dispersion_profils(df, variables):
    dispersion = df.groupby('equipe').agg({v: ['median', 'std', 'min', 'max'] for v in variables})
    return dispersion

#choix des variables que l'on va regarder
#passes :
variables_passes = [
    'pass_count_attempted_p30tip',            #Passes tentees
    'pass_count_completed_p30tip',            #Passes reussies
    'pass_count_linebreak_attempted_p30tip',  #Passes qui cassent une ligne
    'pass_count_difficultpass_attempted_p30tip', #Passes difficiles
    'pass_count_longrange_attempted_p30tip',  #Passes longues
    'pass_count_dangerous_attempted_p30tip',  #Passes dangereuses
    'pass_count_shotwithin10s_p30tip'         #Passes suivies d'un tir sous 10 s
]
variables_qualite = [
    'pass_pct_completed',                     #Taux de passes reussies
    'pass_avgdistance'                        #Distance moyenne des passes
]

#courses :
variables_courses = [
    'offballrun_count_p30tip',                #Toutes les courses sans ballon
    'offballrun_count_dangerous_p30tip',      #Courses dangereuses
    'offballrun_count_targeted_p30tip',       #Courses visees par une passe
    'offballrun_count_received_p30tip',       #Courses avec reception
    'offballrun_count_penaltyarea_p30tip',    #Courses dans la surface
    'offballrun_count_abovehsr_p30tip'        #Courses au-dessus du seuil HSR
]
variables_physiques = [
    'total_metersperminute_full_all',         #Metres parcourus par minute
    'total_metersperminute_full_tip',         #Metres/min en possession
    'total_metersperminute_full_otip',        #Metres/min hors possession
    'hsr_distance_full_all'                   #Distance a haute vitesse
]

#calcul des moyennes pondérées par équipes
tab_passes = moyenne_ponderee_par_equipe(pas, variables_passes, 'poids')
tab_qualite = moyenne_ponderee_par_equipe(pas, variables_qualite, 'poids_passes')
tab_courses = moyenne_ponderee_par_equipe(courses, variables_courses, 'poids')
tab_physique = moyenne_ponderee_par_equipe(phys, variables_physiques, 'poids')

#Types de courses susceptibles d'eclairer le contournement de la zone 14
variables_types = [
    'behindrun_count_p30tip',                 #Courses dans le dos de la defense
    'comingshortrun_count_p30tip',            #Courses en venant court
    'supportrun_count_p30tip',                #Courses de soutien
    'pullingwiderun_count_p30tip',            #Courses ecartant sur l'aile
    'pullinghalfspacerun_count_p30tip'        #Courses vers les demi-espaces
]
tab_types = moyenne_ponderee_par_equipe(courses, variables_types, 'poids') 

#on fusionne les tableaux de moyennes pondérées
equipes = tab_passes.reset_index()
for tableau in [tab_qualite, tab_courses, tab_physique, tab_types]:
    equipes = equipes.merge(tableau.reset_index(), on='equipe', how='outer')

#proportions de passes longues et passes cassant des lignes parmi les passes tentées; courses dangereuses, courses dans le dos de la défense, courses en solution courte parmi les courses
equipes['part_passes_longues'] = (equipes['pass_count_longrange_attempted_p30tip'] /
                                  equipes['pass_count_attempted_p30tip'])
equipes['part_passes_lignes'] = (equipes['pass_count_linebreak_attempted_p30tip'] /
                                 equipes['pass_count_attempted_p30tip'])
equipes['part_courses_dangereuses'] = (equipes['offballrun_count_dangerous_p30tip'] /
                                      equipes['offballrun_count_p30tip'])
equipes['part_courses_dos_defense'] = (equipes['behindrun_count_p30tip'] /
                                       equipes['offballrun_count_p30tip'])
equipes['part_courses_courtes'] = (equipes['comingshortrun_count_p30tip'] /
                                  equipes['offballrun_count_p30tip'])
# %%
#on affiche les stats descriptives sur les passes 
print('\nTOUTES LES EQUIPES : LES PASSES ')
print(equipes[['equipe'] + variables_passes + variables_qualite + ['part_passes_longues', 'part_passes_lignes']])
# %%
#on affiche les stats descriptives sur les courses
print('\nTOUTES LES EQUIPES : COURSES SANS BALLON')
print(equipes[['equipe'] + variables_courses + variables_types + ['part_courses_dangereuses', 'part_courses_dos_defense', 'part_courses_courtes']])

# %%
#on affiche les stats descriptives sur les données physiques
print('\nTOUTES LES EQUIPES : INTENSITE PHYSIQUE')
print(equipes[['equipe'] + variables_physiques])

# %%
#on regarde la distribution entre équipes sur plusieurs stats 
print('\nDISTRIBUTION ENTRE EQUIPES')
variables_clefs = ['pass_count_attempted_p30tip', 'pass_count_linebreak_attempted_p30tip', 'pass_count_difficultpass_attempted_p30tip', 'pass_avgdistance', 'offballrun_count_p30tip', 'offballrun_count_dangerous_p30tip', 'part_passes_longues', 'part_courses_dos_defense']
resume_ligue = equipes[variables_clefs].agg(['mean', 'median', 'std', 'min', 'max'])
print(resume_ligue.round(2))

# %%
#médiane et écart type des profils de joueurs au sein des équipes
print('\nDISPERSION DES PROFILS : PASSES')
print(dispersion_profils(pas, variables_passes).round(2))
print('\nDISPERSION DES PROFILS : COURSES')
print(dispersion_profils(courses, variables_courses).round(2))

# %%
#Focus sur Auckland
#on filtre l'équipe dans une variable et on affiche des stats sur auckland
auckland = equipes.loc[equipes['equipe'] == 'Auckland'].copy()
print('\nAUCKLAND : PROFIL GENERAL')
print(auckland[['equipe'] + variables_clefs].round(2))

# %%
#Comparaison a la moyenne de ligue pour chaque indicateur
print('\nAUCKLAND COMPARE A LA LIGUE')
for variable in variables_clefs:
    valeur = auckland[variable].iloc[0]
    moyenne = equipes[variable].mean()
    mediane = equipes[variable].median()
    ecart_type = equipes[variable].std()
    print('\n', variable)
    print('Auckland :', round(valeur, 2))
    print('Moyenne ligue :', round(moyenne, 2))
    print('Mediane ligue :', round(mediane, 2))
    print('Ecart-type entre equipes :', round(ecart_type, 2))
    print('Ecart Auckland - moyenne :', round(valeur - moyenne, 2))


# %%
#Mediane et ecart type entre les joueurs d'auckland
print('\nDISPERSION DES JOUEURS D\'AUCKLAND : PASSES')
print(dispersion_profils(pas, variables_passes).loc['Auckland'].round(2))
print('\nDISPERSION DES JOUEURS D\'AUCKLAND : COURSES')
print(dispersion_profils(courses, variables_courses).loc['Auckland'].round(2))



# %%
#VISUALISATION 
#Graphique 1 : passes tentees / 30 minutes de possession sur les 13 équipes avec un diagramme en barres
classement = equipes.sort_values('pass_count_attempted_p30tip')
plt.figure(figsize=(10, 6))
plt.barh(classement['equipe'], classement['pass_count_attempted_p30tip'])
plt.xlabel('Passes tentees par 30 min de possession')
plt.title('Volume de passes - 13 equipes')
plt.show()

# %%
#Graphique 2 : passes difficiles / 30 minutes de possession
classement = equipes.sort_values('pass_count_difficultpass_attempted_p30tip')
plt.figure(figsize=(10, 6))
plt.barh(classement['equipe'], classement['pass_count_difficultpass_attempted_p30tip'])
plt.xlabel('Passes difficiles tentees par 30 min de possession')
plt.title('Passes difficiles - 13 equipes')
plt.show()

# %%
#Graphique 3 : courses sans ballon / 30 minutes de possession
classement = equipes.sort_values('offballrun_count_p30tip')
plt.figure(figsize=(10, 6))
plt.barh(classement['equipe'], classement['offballrun_count_p30tip'])
plt.xlabel('Courses sans ballon par 30 min de possession')
plt.title('Volume de courses sans ballon - 13 equipes')
plt.show()

# %%
#Graphique 4 : histogramme des passes tentees des 13 equipes
plt.figure(figsize=(8, 5))
equipes['pass_count_attempted_p30tip'].hist()
plt.xlabel('Passes tentees par 30 min de possession')
plt.ylabel('Nombre d\'equipes')
plt.title('Distribution des volumes de passes entre equipes')
plt.show()

# %%
#Graphique 5 : nuage de points, passes et courses
plt.figure(figsize=(8, 5))
plt.plot(equipes['pass_count_attempted_p30tip'],
         equipes['offballrun_count_p30tip'], 'o', label='Autres equipes et Auckland')
plt.plot(auckland['pass_count_attempted_p30tip'],
         auckland['offballrun_count_p30tip'], 'ro', label='Auckland')
plt.xlabel('Passes tentees / 30 min de possession')
plt.ylabel('Courses sans ballon / 30 min de possession')
plt.title('Relation entre passes et courses sans ballon')
plt.legend()
plt.show()



# %%
