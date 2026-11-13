import pandas as pd
import matplotlib.pyplot as plt


# %%
#on dessine le terrain et la zone 14
#cette fonction est utilisée par les fonctions de visualisation

def tracer_terrain(longueur, largeur):
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    plt.figure(figsize=(11, 6))
    plt.plot(
        [-longueur/2, longueur/2, longueur/2, -longueur/2, -longueur/2],
        [-largeur/2, -largeur/2, largeur/2, largeur/2, -largeur/2], 'k-'
    )
    plt.plot([0, 0], [-largeur/2, largeur/2], 'k-')
    plt.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min], 'r-'
    )
    plt.xlabel('Longueur du terrain (m)')
    plt.ylabel('Largeur du terrain (m)')
    plt.axis('equal')


# %%
#on prépare les informations nécessaires pour chaque match
#on fournit l'identifiant de l'équipe pour fonctionner à domicile ou à l'extérieur

def preparer_match(match_json, evenements, id_equipe):
    longueur = match_json['pitch_length']
    largeur = match_json['pitch_width']
    fin_mi_temps_1 = match_json['match_periods'][0]['end_frame']

    #on conserve les données d'origine sans les modifier
    donnees = evenements.copy()
    donnees['team_id'] = pd.to_numeric(donnees['team_id'], errors='coerce')

    for colonne in ['frame_start', 'frame_end', 'x_start', 'y_start', 'x_end', 'y_end']:
        donnees[colonne] = pd.to_numeric(donnees[colonne], errors='coerce')

    return donnees, longueur, largeur, fin_mi_temps_1


# %%
#on convertit les variables EPV et les indicateurs oui/non des passes
#si une colonne est absente, sa valeur reste manquante

def preparer_qualite_passes(passes):
    variables_qualite_passes = [
        'pass_epv_delta_for',
        'pass_epv_total',
        'pass_reception_epv_for',
        'lead_to_shot',
        'lead_to_goal'
    ]

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

    return passes


# %%
#on affiche les statistiques de qualité de chaque groupe de passes

def afficher_qualite_passes(passes, titre):
    variables = [
        'pass_epv_delta_for', 'pass_epv_total', 'pass_reception_epv_for',
        'lead_to_shot', 'lead_to_goal'
    ]
    print('\n' + titre)
    print('Nombre de passes :', len(passes))
    print('Passes lead_to_shot :', passes['lead_to_shot'].sum())
    print('Proportion lead_to_shot :', round(passes['lead_to_shot'].mean() * 100, 2), '%')
    print('Passes lead_to_goal :', passes['lead_to_goal'].sum())
    print('Proportion lead_to_goal :', round(passes['lead_to_goal'].mean() * 100, 2), '%')
    print('\nQUALITE MOYENNE DES PASSES')
    print(passes[variables].agg(['count', 'mean']).round(4))


# %%
### 1. PASSES DEPUIS LA ZONE 14
#la fonction affiche les résultats et la carte, puis renvoie les passes sélectionnées

def afficher_passes_depuis_zone14(match_json, evenements, id_equipe, nom_match='Match'):
    donnees, longueur, largeur, fin_mi_temps_1 = preparer_match(
        match_json, evenements, id_equipe
    )
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    passes = donnees.loc[
        (donnees['team_id'] == id_equipe)
        & (donnees['event_type'] == 'player_possession')
        & (donnees['end_type'] == 'pass')
    ].copy()

    passes = preparer_qualite_passes(passes)
    afficher_qualite_passes(passes, 'PASSES SUR L ENSEMBLE DU MATCH')

    passes['mi_temps'] = 2
    passes.loc[passes['frame_end'] <= fin_mi_temps_1, 'mi_temps'] = 1

    passes_zone14 = passes.loc[
        (passes['x_end'] >= x_min) & (passes['x_end'] <= x_max)
        & (passes['y_end'] >= y_min) & (passes['y_end'] <= y_max)
    ].copy()

    afficher_qualite_passes(passes_zone14, 'PASSES DEPUIS LA ZONE 14')

    print('\nPASSES DEPUIS LA ZONE 14 PAR JOUEUR')
    print(passes_zone14.groupby('player_name').agg(
        nombre_passes=('event_type', 'size'),
        epv_moyen=('pass_epv_delta_for', 'mean'),
        epv_total_moyen=('pass_epv_total', 'mean'),
        epv_reception_moyen=('pass_reception_epv_for', 'mean'),
        nb_epv_disponibles=('pass_epv_delta_for', 'count'),
        passes_lead_to_shot=('lead_to_shot', 'sum'),
        part_lead_to_shot=('lead_to_shot', 'mean'),
        passes_lead_to_goal=('lead_to_goal', 'sum'),
        part_lead_to_goal=('lead_to_goal', 'mean')
    ).sort_values('nombre_passes', ascending=False).round(4))

    print('\nPASSES DEPUIS LA ZONE 14 PAR MI-TEMPS')
    print(passes_zone14.groupby('mi_temps').agg(
        nombre_passes=('event_type', 'size'),
        epv_moyen=('pass_epv_delta_for', 'mean'),
        epv_total_moyen=('pass_epv_total', 'mean'),
        epv_reception_moyen=('pass_reception_epv_for', 'mean'),
        passes_lead_to_shot=('lead_to_shot', 'sum'),
        part_lead_to_shot=('lead_to_shot', 'mean'),
        passes_lead_to_goal=('lead_to_goal', 'sum'),
        part_lead_to_goal=('lead_to_goal', 'mean')
    ).round(4))

    print('\nEPV moyen de toutes les passes :', round(passes['pass_epv_delta_for'].mean(), 4))
    print('EPV moyen des passes depuis la zone 14 :',
          round(passes_zone14['pass_epv_delta_for'].mean(), 4))
    print('\nPREMIERES PASSES DEPUIS LA ZONE 14')
    print(passes_zone14[['player_name', 'frame_end', 'x_end', 'y_end']].head(10))

    tracer_terrain(longueur, largeur)
    plt.plot(passes_zone14['x_end'], passes_zone14['y_end'], 'bo')
    plt.title(nom_match + ' : passes depuis la zone 14')
    plt.show()
    return passes_zone14


# %%
### 2. FRAPPES DEPUIS LA ZONE 14
#la qualité ici décrit la valeur offensive de la possession, pas un xG

def afficher_frappes_depuis_zone14(match_json, evenements, id_equipe, nom_match='Match'):
    donnees, longueur, largeur, fin_mi_temps_1 = preparer_match(
        match_json, evenements, id_equipe
    )
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    frappes = donnees.loc[
        (donnees['team_id'] == id_equipe)
        & (donnees['event_type'] == 'player_possession')
        & (donnees['end_type'] == 'shot')
    ].copy()

    variables = ['possession_epv_for_end', 'possession_epv_delta_for']
    for variable in variables:
        if variable in frappes.columns:
            frappes[variable] = pd.to_numeric(frappes[variable], errors='coerce')
        else:
            print('Variable absente :', variable)
            frappes[variable] = float('nan')

    print('\nFRAPPES SUR L ENSEMBLE DU MATCH')
    print('Nombre de frappes :', len(frappes))
    print(frappes[variables].agg(['count', 'mean']).round(4))

    frappes['mi_temps'] = 2
    frappes.loc[frappes['frame_end'] <= fin_mi_temps_1, 'mi_temps'] = 1

    frappes_zone14 = frappes.loc[
        (frappes['x_end'] >= x_min) & (frappes['x_end'] <= x_max)
        & (frappes['y_end'] >= y_min) & (frappes['y_end'] <= y_max)
    ].copy()

    print('\nFRAPPES DEPUIS LA ZONE 14')
    print('Nombre de frappes :', len(frappes_zone14))
    print(frappes_zone14[variables].agg(['count', 'mean']).round(4))

    print('\nFRAPPES DEPUIS LA ZONE 14 PAR JOUEUR')
    print(frappes_zone14.groupby('player_name').agg(
        nombre_frappes=('event_type', 'size'),
        epv_fin_moyen=('possession_epv_for_end', 'mean'),
        variation_epv_moyenne=('possession_epv_delta_for', 'mean'),
        nb_epv_disponibles=('possession_epv_for_end', 'count')
    ).sort_values('nombre_frappes', ascending=False).round(4))

    print('\nFRAPPES DEPUIS LA ZONE 14 PAR MI-TEMPS')
    print(frappes_zone14.groupby('mi_temps').agg(
        nombre_frappes=('event_type', 'size'),
        epv_fin_moyen=('possession_epv_for_end', 'mean'),
        variation_epv_moyenne=('possession_epv_delta_for', 'mean')
    ).round(4))

    print('\nPROPORTION DE FRAPPES DEPUIS LA ZONE 14')
    if len(frappes) > 0:
        print(round(len(frappes_zone14) / len(frappes) * 100, 2), '%')
    else:
        print('Aucune frappe')

    print('\nEPV moyen de toutes les frappes :',
          round(frappes['possession_epv_for_end'].mean(), 4))
    print('EPV moyen des frappes depuis la zone 14 :',
          round(frappes_zone14['possession_epv_for_end'].mean(), 4))
    print('\nPREMIERES FRAPPES DEPUIS LA ZONE 14')
    print(frappes_zone14[['player_name', 'frame_end', 'x_end', 'y_end']].head(10))

    tracer_terrain(longueur, largeur)
    plt.plot(frappes_zone14['x_end'], frappes_zone14['y_end'], 'ro')
    plt.title(nom_match + ' : frappes depuis la zone 14')
    plt.show()
    return frappes_zone14


# %%
### 3. PASSES VERS LA ZONE 14
#on estime la réception par la possession suivante d'un coéquipier
#ce rapprochement temporel n'est pas une liaison officielle des événements

def afficher_passes_vers_zone14(match_json, evenements, id_equipe, nom_match='Match'):
    donnees, longueur, largeur, fin_mi_temps_1 = preparer_match(
        match_json, evenements, id_equipe
    )
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    possessions = donnees.loc[
        donnees['event_type'] == 'player_possession'
    ].copy()
    possessions = preparer_qualite_passes(possessions)
    possessions = possessions.sort_values(
        ['frame_start', 'frame_end']
    ).reset_index(drop=True)

    #pour chaque possession, on regarde la possession suivante
    possession_actuelle = possessions.iloc[:-1].reset_index(drop=True).copy()
    possession_suivante = possessions.iloc[1:].reset_index(drop=True).copy()

    possession_actuelle['equipe_suivante'] = possession_suivante['team_id']
    possession_actuelle['joueur_suivant'] = possession_suivante['player_id']
    possession_actuelle['frame_suivante'] = possession_suivante['frame_start']
    possession_actuelle['x_arrivee'] = possession_suivante['x_start']
    possession_actuelle['y_arrivee'] = possession_suivante['y_start']
    possession_actuelle['ecart_frames'] = (
        possession_actuelle['frame_suivante'] - possession_actuelle['frame_end']
    )

    #0 à 30 frames : hypothèse de réception dans un court délai
    passes_receptions = possession_actuelle.loc[
        (possession_actuelle['team_id'] == id_equipe)
        & (possession_actuelle['end_type'] == 'pass')
        & (possession_actuelle['equipe_suivante'] == id_equipe)
        & (possession_actuelle['joueur_suivant'] != possession_actuelle['player_id'])
        & (possession_actuelle['ecart_frames'] >= 0)
        & (possession_actuelle['ecart_frames'] <= 30)
        & ~(
            (possession_actuelle['frame_end'] <= fin_mi_temps_1)
            & (possession_actuelle['frame_suivante'] > fin_mi_temps_1)
        )
    ].copy()

    #on sélectionne une arrivée en zone 14 depuis l'extérieur
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

    print('\nPASSES VERS LA ZONE 14 - RECEPTIONS PROBABLES')
    print('Nombre de passes avec réception probable :', len(passes_receptions))
    afficher_qualite_passes(passes_vers_zone14, 'PASSES ARRIVANT EN ZONE 14')

    print('\nPASSES VERS LA ZONE 14 PAR JOUEUR')
    print(passes_vers_zone14.groupby('player_name').agg(
        nombre_passes=('event_type', 'size'),
        epv_moyen=('pass_epv_delta_for', 'mean'),
        nb_epv_disponibles=('pass_epv_delta_for', 'count'),
        passes_lead_to_shot=('lead_to_shot', 'sum'),
        part_lead_to_shot=('lead_to_shot', 'mean'),
        passes_lead_to_goal=('lead_to_goal', 'sum'),
        part_lead_to_goal=('lead_to_goal', 'mean')
    ).sort_values('nombre_passes', ascending=False).round(4))

    passes_vers_zone14['mi_temps'] = 2
    passes_vers_zone14.loc[
        passes_vers_zone14['frame_end'] <= fin_mi_temps_1, 'mi_temps'
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

    print('\nVERIFICATION DES RECEPTIONS ESTIMEES')
    print(passes_vers_zone14[
        ['player_name', 'frame_end', 'frame_suivante', 'ecart_frames',
         'x_end', 'y_end', 'x_arrivee', 'y_arrivee']
    ].head(10))

    tracer_terrain(longueur, largeur)
    for i in range(len(passes_vers_zone14)):
        passe = passes_vers_zone14.iloc[i]
        plt.arrow(
            passe['x_end'], passe['y_end'],
            passe['x_arrivee'] - passe['x_end'],
            passe['y_arrivee'] - passe['y_end'],
            color='blue', alpha=0.5, head_width=0.9,
            head_length=1.2, length_includes_head=True
        )
    plt.title(nom_match + ' : passes vers la zone 14 (réceptions estimées)')
    plt.show()
    return passes_vers_zone14


# %%
### 4. COURSES SANS BALLON VERS LA ZONE 14

def afficher_courses_vers_zone14(match_json, evenements, id_equipe, nom_match='Match'):
    donnees, longueur, largeur, fin_mi_temps_1 = preparer_match(
        match_json, evenements, id_equipe
    )
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    courses = donnees.loc[
        (donnees['team_id'] == id_equipe)
        & (donnees['event_type'] == 'off_ball_run')
    ].copy()
    courses_vers_zone14 = courses.loc[
        (courses['x_end'] >= x_min) & (courses['x_end'] <= x_max)
        & (courses['y_end'] >= y_min) & (courses['y_end'] <= y_max)
        & ~(
            (courses['x_start'] >= x_min) & (courses['x_start'] <= x_max)
            & (courses['y_start'] >= y_min) & (courses['y_start'] <= y_max)
        )
    ].copy()

    print('\nCOURSES SANS BALLON VERS LA ZONE 14')
    print('Nombre total de courses :', len(courses))
    print('Nombre de courses vers la zone 14 :', len(courses_vers_zone14))
    print('\nCOURSES VERS LA ZONE 14 PAR JOUEUR')
    print(courses_vers_zone14['player_name'].value_counts())
    if 'received' in courses_vers_zone14.columns:
        print('\nRECEPTION DU BALLON')
        print(courses_vers_zone14['received'].value_counts(dropna=False))

    courses_vers_zone14['mi_temps'] = 2
    courses_vers_zone14.loc[
        courses_vers_zone14['frame_end'] <= fin_mi_temps_1, 'mi_temps'
    ] = 1
    print('\nCOURSES VERS LA ZONE 14 PAR MI-TEMPS')
    print(courses_vers_zone14['mi_temps'].value_counts())
    print('\nPREMIERES COURSES')
    print(courses_vers_zone14[
        ['player_name', 'x_start', 'y_start', 'x_end', 'y_end']
    ].head(10))

    tracer_terrain(longueur, largeur)
    for i in range(len(courses_vers_zone14)):
        course = courses_vers_zone14.iloc[i]
        plt.arrow(
            course['x_start'], course['y_start'],
            course['x_end'] - course['x_start'],
            course['y_end'] - course['y_start'],
            color='green', alpha=0.55, head_width=0.9,
            head_length=1.2, length_includes_head=True
        )
    plt.title(nom_match + ' : courses sans ballon vers la zone 14')
    plt.show()
    return courses_vers_zone14


# %%
### 5. COURSES SANS BALLON DEPUIS LA ZONE 14

def afficher_courses_depuis_zone14(match_json, evenements, id_equipe, nom_match='Match'):
    donnees, longueur, largeur, fin_mi_temps_1 = preparer_match(
        match_json, evenements, id_equipe
    )
    x_min = 17.5
    x_max = longueur / 2 - 16.5
    y_min = -11.33
    y_max = 11.33

    courses = donnees.loc[
        (donnees['team_id'] == id_equipe)
        & (donnees['event_type'] == 'off_ball_run')
    ].copy()
    courses_depuis_zone14 = courses.loc[
        (courses['x_start'] >= x_min) & (courses['x_start'] <= x_max)
        & (courses['y_start'] >= y_min) & (courses['y_start'] <= y_max)
        & ~(
            (courses['x_end'] >= x_min) & (courses['x_end'] <= x_max)
            & (courses['y_end'] >= y_min) & (courses['y_end'] <= y_max)
        )
    ].copy()

    print('\nCOURSES SANS BALLON DEPUIS LA ZONE 14')
    print('Nombre total de courses :', len(courses))
    print('Nombre de courses depuis la zone 14 :', len(courses_depuis_zone14))
    print('\nCOURSES DEPUIS LA ZONE 14 PAR JOUEUR')
    print(courses_depuis_zone14['player_name'].value_counts())
    if 'received' in courses_depuis_zone14.columns:
        print('\nRECEPTION DU BALLON')
        print(courses_depuis_zone14['received'].value_counts(dropna=False))

    courses_depuis_zone14['mi_temps'] = 2
    courses_depuis_zone14.loc[
        courses_depuis_zone14['frame_end'] <= fin_mi_temps_1, 'mi_temps'
    ] = 1
    print('\nCOURSES DEPUIS LA ZONE 14 PAR MI-TEMPS')
    print(courses_depuis_zone14['mi_temps'].value_counts())
    print('\nPREMIERES COURSES')
    print(courses_depuis_zone14[
        ['player_name', 'x_start', 'y_start', 'x_end', 'y_end']
    ].head(10))

    tracer_terrain(longueur, largeur)
    for i in range(len(courses_depuis_zone14)):
        course = courses_depuis_zone14.iloc[i]
        plt.arrow(
            course['x_start'], course['y_start'],
            course['x_end'] - course['x_start'],
            course['y_end'] - course['y_start'],
            color='green', alpha=0.55, head_width=0.9,
            head_length=1.2, length_includes_head=True
        )
    plt.title(nom_match + ' : courses sans ballon depuis la zone 14')
    plt.show()
    return courses_depuis_zone14

