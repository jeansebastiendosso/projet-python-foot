# %%
import s3fs
import pandas as pd
import json

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
#on utilise les dictionnaires Python comme dans le code de chargement du JSON
id_auckland = match_1886347['home_team']['id']
longueur = match_1886347['pitch_length']
largeur = match_1886347['pitch_width']

#on récupère la fin de la première mi-temps
fin_mi_temps_1 = match_1886347['match_periods'][0]['end_frame']


### PARTIE SUR LES PASSES DEPUIS LA ZONE 14
# %%
#on sélectionne les passes réalisées par Auckland
#les passes sont des possessions de joueur qui se terminent par une passe
passes = events_1886347.loc[
    (events_1886347['team_id'].astype(int) == id_auckland)
    & (events_1886347['event_type'] == 'player_possession')
    & (events_1886347['end_type'] == 'pass')
].copy()

#on regarde les passes récupérées
print('Nombre de passes :', len(passes))
print(passes[['player_name', 'frame_end', 'x_end', 'y_end']].head())


# %%
#on définit la zone 14 devant la surface adverse
#les coordonnées des événements sont déjà orientées vers le but adverse
#on ne doit donc pas les inverser entre les deux mi-temps
x_min = 17.5
x_max = longueur / 2 - 16.5
y_min = -11.33
y_max = 11.33

#on convertit les coordonnées en nombres décimaux pour éviter l'erreur Arrow
passes['x_end'] = passes['x_end'].astype(float)
passes['y_end'] = passes['y_end'].astype(float)
passes['frame_end'] = passes['frame_end'].astype(float)

#x_end et y_end représentent la fin de la possession du joueur
#comme celle-ci se termine par une passe, on utilise ces coordonnées
#pour identifier le point de départ de la passe

#on sélectionne uniquement les passes dont le départ est dans la zone 14
passes_zone14 = passes.loc[
    (passes['x_end'] >= x_min)
    & (passes['x_end'] <= x_max)
    & (passes['y_end'] >= y_min)
    & (passes['y_end'] <= y_max)
].copy()


# %%
#on regarde le nombre de passes effectuées depuis la zone 14
print('\nPASSES DEPUIS LA ZONE 14')
print('Nombre total de passes :', len(passes_zone14))

#on regarde les joueurs concernés
print(passes_zone14['player_name'].value_counts())

#on regarde les premières passes
print(passes_zone14[
    ['player_name', 'frame_end', 'x_end', 'y_end']
].head(10))


# %%
#on compare le nombre de passes entre les deux mi-temps
passes_zone14['mi_temps'] = 2

passes_zone14.loc[
    passes_zone14['frame_end'] <= fin_mi_temps_1,
    'mi_temps'
] = 1

print('\nPASSES DEPUIS LA ZONE 14 PAR MI-TEMPS')
print(passes_zone14['mi_temps'].value_counts())


# %%
#VISUALISATION
#on représente les points de départ des passes depuis la zone 14
import matplotlib.pyplot as plt

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

# %%
