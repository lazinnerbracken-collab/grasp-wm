# Filtrage des prises Panda par collisions et trajectoires

Une alternative **Coal + RRT-Connect** est maintenant disponible et testee : voir `README_COAL.md`. Le choix `--checker coal` active ce backend ; le backend MuJoCo reste celui par defaut.

Cette extension utilise les 100 predictions Contact-GraspNet deja telechargees dans `panda_grasp_inputs`. Elle tourne localement dans ce dossier. Aucun deploiement sur le serveur SSH n'a ete effectue.

## Choix technique

HPP-FCL / Coal est un moteur de collision, pas un planificateur de trajectoires. Une architecture possible est Pinocchio + Coal + OMPL. Ici, le modele Panda et la scene sont deja dans MuJoCo : on reutilise ses geometries de collision et on ajoute une implementation locale de RRT-Connect en NumPy. Aucune installation de Coal ou d'OMPL n'est necessaire pour cette version.

References : https://github.com/coal-library/coal et https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTConnect.html

## Pipeline

1. Charger les prises, scores et transformations de camera existants. Placer les poses dans le monde de la simulation et dans le repere du site Panda.
2. Trier les prises par score Contact-GraspNet.
3. Calculer l'IK pour une preprise 10 cm en retrait puis pour la prise.
4. Verifier les collisions du robot entier et une marge de 2 mm, doigts ouverts : robot/table, robot/objet, robot/obstacles et autocollisions actives dans le modele.
5. Construire l'approche cartesienne avec des points espaces de 1 cm et verifier chaque interpolation articulaire.
6. Verifier le trajet articulaire direct depuis la posture initiale vers la preprise. S'il est bloque, chercher un detour avec RRT-Connect, puis verifier a nouveau ses segments.
7. Conserver les candidats valides et choisir celui ayant le meilleur score initial.
8. Avec `--execute`, suivre le trajet, fermer la pince avec le controle existant et tenter une levee de 15 cm.

Le contact de fixation entre la base et la table est exempte du filtre. Les contacts doigts/objet ne sont permis pendant l'execution qu'a partir de la fermeture. La levee est surveillee dynamiquement mais n'est PAS preplanifiee avec un objet attache.

## Execution depuis ce dossier

Sous Windows :

```powershell
.\.venv\Scripts\python.exe filter_panda_grasps.py --limit 100 --output collision_results_all --execute
```

Le programme reutilise `panda_predicted_grasp.py`, `panda_lab.py`, `lift_metrics.py`, `assets/` et `panda_grasp_inputs/`. Copier uniquement le nouveau script sur le serveur ne suffit pas. Dans un environnement Python equivalent, la commande est `python filter_panda_grasps.py ...`; les chemins de scene sont regeneres a chaque execution. Linux utilise EGL par defaut pour le rendu.

Pour les graphiques, installer Matplotlib dans le meme environnement puis executer :

```powershell
.\.venv\Scripts\python.exe benchmark_panda_obstacle.py
.\.venv\Scripts\python.exe visualize_collision_results.py
.\.venv\Scripts\python.exe -m unittest test_collision_planning.py
```

Le benchmark utilise `collision_results/scene.xml` et le chemin du candidat 5 dans ce dossier. Les generer au besoin avec `filter_panda_grasps.py --limit 10 --output collision_results`. Il ajoute volontairement un obstacle sur un trajet direct. C'est un test cinematique distinct, pas une demonstration de transport dynamique autour de cet obstacle.

## Fichiers de sortie

- `filter_report.json` : decision et motif pour chaque candidat, contacts bloquants, statistiques du planificateur et eventuel resultat d'execution.
- `filtered_grasps.npz` : indices source, poses dans le monde au site Panda, scores, ouvertures predites et identifiants des candidats retenus. Ce repere differe de celui du fichier d'entree ; la cle est donc `poses_world_site`.
- `candidate_<index>_path.npz` : configurations articulaires de transit et d'approche, pose cible.
- `selected_open_gripper.png` : posture retenue avec pince ouverte.
- `planned_lift.mp4`, `execution.json`, `telemetry.json` : execution physique optionnelle et mesures.

Les indices sont ceux du tableau Python, donc l'indice 5 est le sixieme candidat. Une prise retenue par le filtre n'est pas automatiquement une prise physiquement stable.

## Resultats observes

Sur cette scene de moutarde, pour 100 candidats : 12 retenus, 10 rejetes pour collision ou marge insuffisante, 69 echecs d'IK a la preprise et 9 a la prise. Le candidat 5, meilleur score parmi les retenus, leve l'objet d'environ 14,5 cm et le maintient pendant environ 3 secondes.

Le test avec obstacle artificiel detecte un trajet direct bloque, puis trouve un detour RRT-Connect. Ses segments sont reverifies avec un pas de 0,005 rad. Les six tests couvrent trajet direct, detour, objectif invalide, absence de solution dans le budget, geometries Panda reelles et rejet a distance positive inferieure a la marge.

## Limites

- Le controle est discret : pas articulaire maximal de 0,01 rad par joint par defaut. Ce n'est pas une preuve de collision continue entre les echantillons.
- Les collisions utilisent les geometries, masques et exclusions du MJCF. Le maillage de collision du pot est convexe ; ce n'est pas une verification exacte de toutes les surfaces visuelles.
- Une seule initialisation d'IK est essayee par etape. Un echec d'IK n'etablit pas que la pose est inatteignable ; une recherche multi-initialisations pourrait conserver davantage de prises.
- Un echec RRT dans le budget imparti n'etablit pas l'absence de chemin.
- La camera, l'objet et la scene doivent etre correctement alignes. Les obstacles non modelises ne sont pas detectes.
- Les verifications de planification a 2 mm et la surveillance dynamique avec tolerance de penetration de 2 mm sont deux criteres distincts. Le suivi dynamique peut s'ecarter du chemin articulaire nominal.
- Ce prototype est valide en simulation et ne constitue pas une validation de securite pour le robot reel.
