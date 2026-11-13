# %%
import s3fs
import pandas as pd
import json
import matplotlib.pyplot as plt

fs = s3fs.S3FileSystem(
    endpoint_url="https://minio.lab.sspcloud.fr",
    client_kwargs={"region_name": "us-east-1"},
)
#1886347 correspond au match Auckland vs Newcastle jets
events_1886347 = pd.read_parquet("s3://lfaivre/diffusion/1886347_dynamic_events.parquet", filesystem=fs)

with fs.open("s3://lfaivre/diffusion/1886347_match.json", "r") as fichier:
    match_1886347 = json.load(fichier)

# %%
#on récupère les informations du terrain et d'Auckland dans le JSON
id_auckland = match_1886347['home_team']['id']
longueur = match_1886347['pitch_length']
largeur = match_1886347['pitch_width']

#on récupère la fin de la première mi-temps
fin_mi_temps_1 = match_1886347['match_periods'][0]['end_frame']

#on définit la zone 14 devant la surface adverse
#les coordonnées des événements semblent déjà orientées vers le but adverse
x_min = 17.5
x_max = longueur / 2 - 16.5
y_min = -11.33
y_max = 11.33


### PARTIE SUR LES PASSES DEPUIS LA ZONE 14
# %%
#on sélectionne les passes réalisées par Auckland
#les passes sont des possessions de joueur qui se terminent par une passe
passes = events_1886347.loc[
    (events_1886347['team_id'].astype(int) == id_auckland)
    & (events_1886347['event_type'] == 'player_possession')
    & (events_1886347['end_type'] == 'pass')
].copy()

#on convertit les coordonnées et les variables de qualité en nombres
passes['x_end'] = passes['x_end'].astype(float)
passes['y_end'] = passes['y_end'].astype(float)
passes['frame_end'] = passes['frame_end'].astype(float)

#on sélectionne les indicateurs de qualité des passes
#pass_epv_delta_for : augmentation de la valeur offensive attendue
#pass_epv_total : contribution nette attendue de la passe
#pass_reception_epv_for : valeur offensive attendue à la réception
#lead_to_shot : indique si l'événement est lié à un tir ensuite
#lead_to_goal : indique si l'événement est lié à un but ensuite
variables_qualite_passes = [
    'pass_epv_delta_for',
    'pass_epv_total',
    'pass_reception_epv_for',
    'lead_to_shot',
    'lead_to_goal'
]

#on vérifie que les variables sont disponibles et on les convertit
#les indicateurs lead_to_shot et lead_to_goal sont des vrais/faux
#1 = oui, 0 = non ; leur moyenne donne une proportion
for variable in variables_qualite_passes:
    if variable in passes.columns:
        if variable in ['lead_to_shot', 'lead_to_goal']:
            passes[variable] = passes[variable].astype(str).str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0}
            )
        else:
            passes[variable] = pd.to_numeric(passes[variable], errors='coerce')
    else:
        print('Variable absente :', variable)
        passes[variable] = float('nan')

#on regarde le nombre de passes et leur qualité moyenne
print('\nPASSES D AUCKLAND SUR LE MATCH')
print('Nombre de passes :', len(passes))
#on compte les passes liées à un tir ou à un but
print('Passes lead_to_shot :', passes['lead_to_shot'].sum())
print('Proportion lead_to_shot :', round(passes['lead_to_shot'].mean() * 100, 2), '%')
print('Passes lead_to_goal :', passes['lead_to_goal'].sum())
print('Proportion lead_to_goal :', round(passes['lead_to_goal'].mean() * 100, 2), '%')


print('\nQUALITE MOYENNE DES PASSES')
print(passes[variables_qualite_passes].agg(['count', 'mean']).round(4))

#on regarde les premières passes récupérées
print(passes[
    ['player_name', 'frame_end', 'x_end', 'y_end']
    + variables_qualite_passes
].head())


# %%
#on distingue les deux mi-temps pour toutes les passes
passes['mi_temps'] = 2

passes.loc[
    passes['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

#on sélectionne uniquement les passes dont le départ est dans la zone 14
#x_end et y_end correspondent à la fin de la possession du joueur
passes_zone14 = passes.loc[
    (passes['x_end'] >= x_min)
    & (passes['x_end'] <= x_max)
    & (passes['y_end'] >= y_min)
    & (passes['y_end'] <= y_max)
].copy()


# %%
#on regarde le nombre de passes effectuées depuis la zone 14
#et leur qualité moyenne
print('\nPASSES DEPUIS LA ZONE 14')
print('Nombre total de passes :', len(passes_zone14))
#on compte les passes liées à un tir ou à un but
print('Passes lead_to_shot :', passes_zone14['lead_to_shot'].sum())
print('Proportion lead_to_shot :', round(passes_zone14['lead_to_shot'].mean() * 100, 2), '%')
print('Passes lead_to_goal :', passes_zone14['lead_to_goal'].sum())
print('Proportion lead_to_goal :', round(passes_zone14['lead_to_goal'].mean() * 100, 2), '%')


print('\nQUALITE MOYENNE DES PASSES DEPUIS LA ZONE 14')
print(
    passes_zone14[variables_qualite_passes]
    .agg(['count', 'mean']).round(4)
)

#on regarde les joueurs concernés avec leur nombre de passes
#et la valeur offensive moyenne de leurs passes
print('\nPASSES DEPUIS LA ZONE 14 PAR JOUEUR')

qualite_joueurs_passes = passes_zone14.groupby('player_name').agg(
    nombre_passes=('event_type', 'size'),
    epv_moyen=('pass_epv_delta_for', 'mean'),
    epv_total_moyen=('pass_epv_total', 'mean'),
    epv_reception_moyen=('pass_reception_epv_for', 'mean'),
    nb_epv_disponibles=('pass_epv_delta_for', 'count'),
    passes_lead_to_shot=('lead_to_shot', 'sum'),
    part_lead_to_shot=('lead_to_shot', 'mean'),
    passes_lead_to_goal=('lead_to_goal', 'sum'),
    part_lead_to_goal=('lead_to_goal', 'mean')
)

print(
    qualite_joueurs_passes.sort_values(
        'nombre_passes', ascending=False
    ).round(4)
)

#on regarde les premières passes
print('\nPREMIERES PASSES DEPUIS LA ZONE 14')
print(passes_zone14[
    ['player_name', 'frame_end', 'x_end', 'y_end']
    + variables_qualite_passes
].head(10))


# %%
#on compare le nombre de passes et leur qualité entre les deux mi-temps
print('\nPASSES DEPUIS LA ZONE 14 PAR MI-TEMPS')

qualite_mi_temps_passes = passes_zone14.groupby('mi_temps').agg(
    nombre_passes=('event_type', 'size'),
    epv_moyen=('pass_epv_delta_for', 'mean'),
    epv_total_moyen=('pass_epv_total', 'mean'),
    epv_reception_moyen=('pass_reception_epv_for', 'mean'),
    nb_epv_disponibles=('pass_epv_delta_for', 'count'),
    passes_lead_to_shot=('lead_to_shot', 'sum'),
    part_lead_to_shot=('lead_to_shot', 'mean'),
    passes_lead_to_goal=('lead_to_goal', 'sum'),
    part_lead_to_goal=('lead_to_goal', 'mean')
)

print(qualite_mi_temps_passes.round(4))


# %%
#on compare la qualité des passes depuis la zone 14
#avec la qualité des passes sur l'ensemble du match

print('\nCOMPARAISON QUALITE DES PASSES')

print('EPV offensif moyen de toutes les passes :',
      round(passes['pass_epv_delta_for'].mean(), 4))

print('EPV offensif moyen des passes depuis la zone 14 :',
      round(passes_zone14['pass_epv_delta_for'].mean(), 4))


# %%
#VISUALISATION
#on représente les points de départ des passes depuis la zone 14

plt.figure(figsize=(11, 6))

#on dessine les limites extérieures du terrain
plt.plot(
    [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
    [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2],
    'k-'
)

#on dessine la ligne médiane
plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')

#on représente les limites de la zone 14
plt.plot(
    [x_min, x_max, x_max, x_min, x_min],
    [y_min, y_min, y_max, y_max, y_min],
    'r-'
)

#on place les points de départ des passes
plt.plot(
    passes_zone14['x_end'],
    passes_zone14['y_end'],
    'bo'
)

plt.xlabel('Longueur du terrain (m)')
plt.ylabel('Largeur du terrain (m)')
plt.title('Auckland - Newcastle : passes depuis la zone 14')

plt.axis('equal')
plt.show()


### PARTIE SUR LES FRAPPES DEPUIS LA ZONE 14
# %%
#on sélectionne les frappes réalisées par Auckland
#les frappes sont des possessions de joueur qui se terminent par un tir
frappes = events_1886347.loc[
    (events_1886347['team_id'].astype(int) == id_auckland)
    & (events_1886347['event_type'] == 'player_possession')
    & (events_1886347['end_type'] == 'shot')
].copy()

#on convertit les coordonnées en nombres
frappes['x_end'] = frappes['x_end'].astype(float)
frappes['y_end'] = frappes['y_end'].astype(float)
frappes['frame_end'] = frappes['frame_end'].astype(float)

#on sélectionne les indicateurs EPV associés aux possessions terminées par un tir
#possession_epv_for_end : valeur offensive attendue en fin de possession
#possession_epv_delta_for : variation de valeur offensive pendant la possession
#ces indicateurs ne sont pas des xG
variables_qualite_frappes = [
    'possession_epv_for_end',
    'possession_epv_delta_for'
]

#on vérifie que les variables sont disponibles et on les convertit
for variable in variables_qualite_frappes:
    if variable in frappes.columns:
        frappes[variable] = pd.to_numeric(frappes[variable], errors='coerce')
    else:
        print('Variable absente :', variable)
        frappes[variable] = float('nan')

#on regarde le nombre de frappes récupérées
print('\nFRAPPES D AUCKLAND SUR LE MATCH')
print('Nombre de frappes :', len(frappes))

#on regarde la valeur offensive moyenne des possessions terminées par un tir
print('\nVALEUR OFFENSIVE MOYENNE DES POSSESSIONS TERMINEES PAR UN TIR')
print(
    frappes[variables_qualite_frappes]
    .agg(['count', 'mean']).round(4)
)

print(frappes[
    ['player_name', 'frame_end', 'x_end', 'y_end']
    + variables_qualite_frappes
].head())


# %%
#on distingue les deux mi-temps pour toutes les frappes
frappes['mi_temps'] = 2

frappes.loc[
    frappes['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

#on sélectionne uniquement les frappes dont le départ est dans la zone 14
frappes_zone14 = frappes.loc[
    (frappes['x_end'] >= x_min)
    & (frappes['x_end'] <= x_max)
    & (frappes['y_end'] >= y_min)
    & (frappes['y_end'] <= y_max)
].copy()


# %%
#on regarde le nombre de frappes effectuées depuis la zone 14
print('\nFRAPPES DEPUIS LA ZONE 14')
print('Nombre total de frappes :', len(frappes_zone14))

#on regarde la valeur offensive moyenne de ces possessions
print('\nVALEUR OFFENSIVE MOYENNE DES POSSESSIONS TERMINEES PAR UN TIR DEPUIS LA ZONE 14')
print(
    frappes_zone14[variables_qualite_frappes]
    .agg(['count', 'mean']).round(4)
)

#on regarde les joueurs concernés avec le nombre de frappes
#et les indicateurs EPV moyens
print('\nFRAPPES DEPUIS LA ZONE 14 PAR JOUEUR')

qualite_joueurs_frappes = frappes_zone14.groupby('player_name').agg(
    nombre_frappes=('event_type', 'size'),
    epv_fin_moyen=('possession_epv_for_end', 'mean'),
    variation_epv_moyenne=('possession_epv_delta_for', 'mean'),
    nb_epv_disponibles=('possession_epv_for_end', 'count')
)

print(
    qualite_joueurs_frappes.sort_values(
        'nombre_frappes', ascending=False
    ).round(4)
)

#on regarde les premières frappes
print('\nPREMIERES FRAPPES DEPUIS LA ZONE 14')
print(frappes_zone14[
    ['player_name', 'frame_end', 'x_end', 'y_end']
    + variables_qualite_frappes
].head(10))


# %%
#on compare le nombre de frappes et les indicateurs EPV
#entre les deux mi-temps
print('\nFRAPPES DEPUIS LA ZONE 14 PAR MI-TEMPS')

qualite_mi_temps_frappes = frappes_zone14.groupby('mi_temps').agg(
    nombre_frappes=('event_type', 'size'),
    epv_fin_moyen=('possession_epv_for_end', 'mean'),
    variation_epv_moyenne=('possession_epv_delta_for', 'mean'),
    nb_epv_disponibles=('possession_epv_for_end', 'count')
)

print(qualite_mi_temps_frappes.round(4))


# %%
#on calcule la proportion de frappes partant de la zone 14
#par rapport à l'ensemble des frappes d'Auckland
print('\nPROPORTION DE FRAPPES DEPUIS LA ZONE 14')

print('Nombre total de frappes Auckland :', len(frappes))
print('Nombre de frappes depuis la zone 14 :', len(frappes_zone14))

if len(frappes) > 0:
    proportion_frappes = len(frappes_zone14) / len(frappes) * 100
    print('Proportion de frappes depuis la zone 14 :',
          round(proportion_frappes, 2), '%')


# %%
#on compare les indicateurs EPV des possessions terminées par un tir
#depuis la zone 14 et sur l'ensemble du match
print('\nCOMPARAISON VALEUR OFFENSIVE DES POSSESSIONS TERMINEES PAR UN TIR')

print('EPV moyen en fin de possession - toutes les frappes :',
      round(frappes['possession_epv_for_end'].mean(), 4))

print('EPV moyen en fin de possession - frappes zone 14 :',
      round(frappes_zone14['possession_epv_for_end'].mean(), 4))


# %%
#VISUALISATION
#on représente les points de départ des frappes depuis la zone 14
plt.figure(figsize=(11, 6))

#on dessine les limites extérieures du terrain
plt.plot(
    [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
    [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2],
    'k-'
)

#on dessine la ligne médiane
plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')

#on représente les limites de la zone 14
plt.plot(
    [x_min, x_max, x_max, x_min, x_min],
    [y_min, y_min, y_max, y_max, y_min],
    'r-'
)

#on place les points de départ des frappes
plt.plot(
    frappes_zone14['x_end'],
    frappes_zone14['y_end'],
    'ro'
)

plt.xlabel('Longueur du terrain (m)')
plt.ylabel('Largeur du terrain (m)')
plt.title('Auckland - Newcastle : frappes depuis la zone 14')

plt.axis('equal')
plt.show()

# %%

### PARTIE SUR LES PASSES VERS LA ZONE 14
# %%
#on cherche le point d'arrivée des passes dans la possession suivante
#dans SkillCorner, x_end et y_end sont la fin de la possession du passeur,
#pas directement le point de réception de sa passe
#cette méthode ne retrouve que des réceptions probables, et non toutes les passes
possessions = events_1886347.loc[
    events_1886347['event_type'] == 'player_possession'
].copy()

#on convertit les colonnes nécessaires en nombres
possessions['team_id'] = possessions['team_id'].astype(int)
possessions['frame_start'] = pd.to_numeric(possessions['frame_start'], errors='coerce')
possessions['frame_end'] = pd.to_numeric(possessions['frame_end'], errors='coerce')

for coordonnee in ['x_start', 'y_start', 'x_end', 'y_end']:
    possessions[coordonnee] = pd.to_numeric(possessions[coordonnee], errors='coerce')

#on convertit les indicateurs de qualité des passes pour calculer les moyennes
#lead_to_shot et lead_to_goal sont des indicateurs oui/non
for variable in variables_qualite_passes:
    if variable in possessions.columns:
        if variable in ['lead_to_shot', 'lead_to_goal']:
            possessions[variable] = possessions[variable].astype(str).str.lower().map(
                {'true': 1, 'false': 0, '1': 1, '0': 0}
            )
        else:
            possessions[variable] = pd.to_numeric(possessions[variable], errors='coerce')
    else:
        print('Variable absente :', variable)
        possessions[variable] = float('nan')

#on classe les possessions de tous les joueurs dans l'ordre du match
possessions = possessions.sort_values(['frame_start', 'frame_end']).reset_index(drop=True)

#pour chaque possession, on récupère les informations de la suivante
#cette association sert à identifier une réception probable de la passe
possession_actuelle = possessions.iloc[:-1].reset_index(drop=True).copy()
possession_suivante = possessions.iloc[1:].reset_index(drop=True).copy()

possession_actuelle['equipe_suivante'] = possession_suivante['team_id']
possession_actuelle['joueur_suivant'] = possession_suivante['player_id']
possession_actuelle['frame_suivante'] = possession_suivante['frame_start']
possession_actuelle['x_arrivee'] = possession_suivante['x_start']
possession_actuelle['y_arrivee'] = possession_suivante['y_start']

#on mesure le temps entre la passe et la possession suivante
#les frames sont enregistrées à 10 images par seconde
possession_actuelle['ecart_frames'] = (
    possession_actuelle['frame_suivante'] - possession_actuelle['frame_end']
)

#on conserve les passes d'Auckland qui semblent reçues par un coéquipier
#dans un délai de 0 à 30 frames (3 secondes)
#il s'agit d'un rapprochement temporel, pas d'une liaison officielle des événements
passes_receptions = possession_actuelle.loc[
    (possession_actuelle['team_id'] == id_auckland)
    & (possession_actuelle['end_type'] == 'pass')
    & (possession_actuelle['equipe_suivante'] == id_auckland)
    & (possession_actuelle['joueur_suivant'] != possession_actuelle['player_id'])
    & (possession_actuelle['ecart_frames'] >= 0)
    & (possession_actuelle['ecart_frames'] <= 30)
].copy()

#on conserve les passes dont la réception estimée est en zone 14
#et dont le départ est en dehors de cette zone
passes_vers_zone14 = passes_receptions.loc[
    (passes_receptions['x_arrivee'] >= x_min)
    & (passes_receptions['x_arrivee'] <= x_max)
    & (passes_receptions['y_arrivee'] >= y_min)
    & (passes_receptions['y_arrivee'] <= y_max)
    & ~(
        (passes_receptions['x_end'] >= x_min)
        & (passes_receptions['x_end'] <= x_max)
        & (passes_receptions['y_end'] >= y_min)
        & (passes_receptions['y_end'] <= y_max)
    )
].copy()

# %%
#on regarde le nombre de passes vers la zone 14 et leur qualité
print('\nPASSES VERS LA ZONE 14 - RECEPTIONS PROBABLES')
print('Nombre de passes d Auckland :', len(passes))
print('Nombre de passes avec une réception probable identifiée :', len(passes_receptions))
print('Nombre de passes arrivant dans la zone 14 depuis l extérieur :', len(passes_vers_zone14))
#on compte les passes liées à un tir ou à un but
print('Passes lead_to_shot :', passes_vers_zone14['lead_to_shot'].sum())
print('Proportion lead_to_shot :', round(passes_vers_zone14['lead_to_shot'].mean() * 100, 2), '%')
print('Passes lead_to_goal :', passes_vers_zone14['lead_to_goal'].sum())
print('Proportion lead_to_goal :', round(passes_vers_zone14['lead_to_goal'].mean() * 100, 2), '%')


print('\nQUALITE MOYENNE DES PASSES VERS LA ZONE 14')
print(passes_vers_zone14[variables_qualite_passes].agg(['count', 'mean']).round(4))

print('\nPASSES VERS LA ZONE 14 PAR JOUEUR')
print(passes_vers_zone14.groupby('player_name').agg(
    nombre_passes=('event_type', 'size'),
    epv_moyen=('pass_epv_delta_for', 'mean'),
    nb_epv_disponibles=('pass_epv_delta_for', 'count'),
    passes_lead_to_shot=('lead_to_shot', 'sum'),
    part_lead_to_shot=('lead_to_shot', 'mean'),
    passes_lead_to_goal=('lead_to_goal', 'sum'),
    part_lead_to_goal=('lead_to_goal', 'mean')
).round(4))

#on compare les deux mi-temps
passes_vers_zone14['mi_temps'] = 2
passes_vers_zone14.loc[
    passes_vers_zone14['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

print('\nPASSES VERS LA ZONE 14 PAR MI-TEMPS')
print(passes_vers_zone14.groupby('mi_temps').agg(
    nombre_passes=('event_type', 'size'),
    epv_moyen=('pass_epv_delta_for', 'mean'),
    passes_lead_to_shot=('lead_to_shot', 'sum'),
    part_lead_to_shot=('lead_to_shot', 'mean'),
    passes_lead_to_goal=('lead_to_goal', 'sum'),
    part_lead_to_goal=('lead_to_goal', 'mean')
).round(4))

#on contrôle quelques associations passeur / réception estimée
print('\nVERIFICATION DES RECEPTIONS ESTIMEES')
print(passes_vers_zone14[
    ['player_name', 'frame_end', 'frame_suivante', 'ecart_frames',
     'x_end', 'y_end', 'x_arrivee', 'y_arrivee']
].head(10))

# %%
#VISUALISATION DES PASSES VERS LA ZONE 14
plt.figure(figsize=(11, 6))
plt.plot(
    [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
    [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2], 'k-'
)
plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], 'r-')

#on dessine une flèche de chaque départ de passe vers la réception estimée
for i in range(len(passes_vers_zone14)):
    passe = passes_vers_zone14.iloc[i]
    plt.arrow(
        passe['x_end'], passe['y_end'],
        passe['x_arrivee'] - passe['x_end'],
        passe['y_arrivee'] - passe['y_end'],
        color='blue', alpha=0.5, head_width=0.9,
        head_length=1.2, length_includes_head=True
    )

plt.xlabel('Longueur du terrain (m)')
plt.ylabel('Largeur du terrain (m)')
plt.title('Auckland - Newcastle : passes vers la zone 14 (réceptions estimées)')
plt.axis('equal')
plt.show()


### PARTIE SUR LES COURSES VERS LA ZONE 14
# %%
#on sélectionne les courses sans ballon d'Auckland
#pour une course, x_start et y_start sont le départ ; x_end et y_end sont l'arrivée
courses_match = events_1886347.loc[
    (events_1886347['team_id'].astype(int) == id_auckland)
    & (events_1886347['event_type'] == 'off_ball_run')
].copy()

#on convertit les coordonnées en nombres décimaux
for coordonnee in ['x_start', 'y_start', 'x_end', 'y_end', 'frame_end']:
    courses_match[coordonnee] = pd.to_numeric(courses_match[coordonnee], errors='coerce')

#on sélectionne les courses qui arrivent en zone 14 depuis l'extérieur
courses_vers_zone14 = courses_match.loc[
    (courses_match['x_end'] >= x_min)
    & (courses_match['x_end'] <= x_max)
    & (courses_match['y_end'] >= y_min)
    & (courses_match['y_end'] <= y_max)
    & ~(
        (courses_match['x_start'] >= x_min)
        & (courses_match['x_start'] <= x_max)
        & (courses_match['y_start'] >= y_min)
        & (courses_match['y_start'] <= y_max)
    )
].copy()

# %%
#on regarde les courses vers la zone 14 et les joueurs concernés
print('\nCOURSES SANS BALLON VERS LA ZONE 14')
print('Nombre total de courses sans ballon d Auckland :', len(courses_match))
print('Nombre de courses arrivant en zone 14 depuis l extérieur :', len(courses_vers_zone14))

print('\nCOURSES VERS LA ZONE 14 PAR JOUEUR')
print(courses_vers_zone14['player_name'].value_counts())

#on regarde la part des courses ayant reçu le ballon, si l'indicateur existe
if 'received' in courses_vers_zone14.columns:
    print('\nCOURSES VERS LA ZONE 14 : RECEPTION DU BALLON')
    print(courses_vers_zone14['received'].value_counts(dropna=False))

#on distingue les deux mi-temps
courses_vers_zone14['mi_temps'] = 2
courses_vers_zone14.loc[
    courses_vers_zone14['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

print('\nCOURSES VERS LA ZONE 14 PAR MI-TEMPS')
print(courses_vers_zone14['mi_temps'].value_counts())

print('\nPREMIERES COURSES VERS LA ZONE 14')
print(courses_vers_zone14[
    ['player_name', 'x_start', 'y_start', 'x_end', 'y_end']
].head(10))

# %%
#VISUALISATION DES COURSES VERS LA ZONE 14
plt.figure(figsize=(11, 6))
plt.plot(
    [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
    [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2], 'k-'
)
plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')
plt.plot([x_min, x_max, x_max, x_min, x_min],
         [y_min, y_min, y_max, y_max, y_min], 'r-')

#on trace une flèche du début vers la fin de chaque course
for i in range(len(courses_vers_zone14)):
    course = courses_vers_zone14.iloc[i]
    plt.arrow(
        course['x_start'], course['y_start'],
        course['x_end'] - course['x_start'],
        course['y_end'] - course['y_start'],
        color='green', alpha=0.55, head_width=0.9,
        head_length=1.2, length_includes_head=True
    )

plt.xlabel('Longueur du terrain (m)')
plt.ylabel('Largeur du terrain (m)')
plt.title('Auckland - Newcastle : courses sans ballon vers la zone 14')
plt.axis('equal')
plt.show()


### PARTIE SUR LES COURSES DEPUIS LA ZONE 14
# %%
#on utilise les courses sans ballon d'Auckland déjà récupérées dans courses_match
#pour une course, x_start et y_start sont le départ ; x_end et y_end sont l'arrivée
#on sélectionne celles qui partent de la zone 14 et se terminent à l'extérieur
courses_depuis_zone14 = courses_match.loc[
    (courses_match['x_start'] >= x_min)
    & (courses_match['x_start'] <= x_max)
    & (courses_match['y_start'] >= y_min)
    & (courses_match['y_start'] <= y_max)
    & ~(
        (courses_match['x_end'] >= x_min)
        & (courses_match['x_end'] <= x_max)
        & (courses_match['y_end'] >= y_min)
        & (courses_match['y_end'] <= y_max)
    )
].copy()


# %%
#on regarde les courses depuis la zone 14 et les joueurs concernés
print('\nCOURSES SANS BALLON DEPUIS LA ZONE 14')
print('Nombre total de courses sans ballon d Auckland :', len(courses_match))
print('Nombre de courses partant de la zone 14 vers l extérieur :', len(courses_depuis_zone14))

print('\nCOURSES DEPUIS LA ZONE 14 PAR JOUEUR')
print(courses_depuis_zone14['player_name'].value_counts())

#on regarde la part des courses ayant reçu le ballon, si l'indicateur existe
if 'received' in courses_depuis_zone14.columns:
    print('\nCOURSES DEPUIS LA ZONE 14 : RECEPTION DU BALLON')
    print(courses_depuis_zone14['received'].value_counts(dropna=False))

#on distingue les deux mi-temps
courses_depuis_zone14['mi_temps'] = 2

courses_depuis_zone14.loc[
    courses_depuis_zone14['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

print('\nCOURSES DEPUIS LA ZONE 14 PAR MI-TEMPS')
print(courses_depuis_zone14['mi_temps'].value_counts())

print('\nPREMIERES COURSES DEPUIS LA ZONE 14')
print(courses_depuis_zone14[
    ['player_name', 'x_start', 'y_start', 'x_end', 'y_end']
].head(10))


# %%
#VISUALISATION DES COURSES DEPUIS LA ZONE 14
plt.figure(figsize=(11, 6))

#on dessine les limites extérieures du terrain
plt.plot(
    [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
    [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2], 'k-'
)

#on dessine la ligne médiane
plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')

#on représente les limites de la zone 14
plt.plot(
    [x_min, x_max, x_max, x_min, x_min],
    [y_min, y_min, y_max, y_max, y_min], 'r-'
)

#on trace une flèche du début vers la fin de chaque course
for i in range(len(courses_depuis_zone14)):
    course = courses_depuis_zone14.iloc[i]

    plt.arrow(
        course['x_start'], course['y_start'],
        course['x_end'] - course['x_start'],
        course['y_end'] - course['y_start'],
        color='green', alpha=0.55, head_width=0.9,
        head_length=1.2, length_includes_head=True
    )

plt.xlabel('Longueur du terrain (m)')
plt.ylabel('Largeur du terrain (m)')
plt.title('Auckland - Newcastle : courses sans ballon depuis la zone 14')

plt.axis('equal')
plt.show()

