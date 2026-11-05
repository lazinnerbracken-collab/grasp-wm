# Coal + RRT-Connect : essai Panda

Version testee sur ce PC : Coal 3.0.4, Python 3.12, MuJoCo 3.3.7 dans un environnement Pixi separe (`coal_env`). Le solveur de dependances couvre Windows et Linux ; seule l'execution Windows a ete testee ici. Rien n'a ete installe sur le serveur SSH.

## Relancer

Ouvrir un terminal dans ce dossier, puis :

```powershell
pixi run --manifest-path coal_env\pixi.toml python filter_panda_grasps.py --checker coal --limit 100 --output coal_results_all --execute
```

Si `pixi` est deja dans votre PATH, remplacer `pixi` par `pixi`. Les assets et observations existants (`assets/`, `panda_grasp_inputs/`) doivent rester a cote des scripts. Le premier lancement sur une nouvelle machine installe les dependances de l'environnement. `coal_env/pixi.lock` conserve les versions resolues.

Le choix `--checker mujoco` conserve le backend initial. Sans `--execute`, on filtre les prises sans effectuer la fermeture/levee physique. Utiliser un dossier de sortie distinct pour chaque experience.

## Qui fait quoi ?

- Contact-GraspNet : fournit les prises deja calculees et leurs scores.
- IK existante et cinematique MuJoCo : fournissent les angles articulaires et positions des liens.
- **Coal : effectue les requetes de collision et de marge pour la planification.** Le backend appelle `mj_kinematics` uniquement pour mettre a jour les transformations ; il ne lit pas les contacts MuJoCo et n'appelle pas `mj_collision` ou `mj_forward` pour decider la validite.
- RRT-Connect : implementation NumPy existante, recherche un detour si le trajet direct est bloque.
- MuJoCo : simule la fermeture et la levee ; la surveillance des contacts pendant cette execution physique reste celle de MuJoCo.

Cette premiere integration n'utilise pas Pinocchio. Coal est reellement utilise, mais le chargement du modele et la cinematique restent dependants de MuJoCo. Ce n'est pas encore un pilote pour le robot reel.

## Conversion des geometries

`coal_collision.py` convertit les geometries compilees du MJCF en formes Coal. Les tailles, echelles et transformations sont donc celles de la scene existante. Les maillages deviennent des enveloppes convexes, comme les collisions du modele MuJoCo actuel. Les plans deviennent des demi-espaces.

Le backend reproduit les masques, exclusions, corps soudes et filtrage parent/enfant du modele, puis exempte la fixation base/table. La scene testee contient 338 paires pertinentes. Les formes non prises en charge et les paires explicites/flexes provoquent une erreur, au lieu d'etre silencieusement ignorees. Le module est destine a cette scene Panda rigide, pas a tous les modeles MuJoCo possibles.

La marge Coal est `security_margin = 0.002` metre. Une separation positive inferieure a 2 mm est rejetee. La distance du contact est enregistree sans soustraire cette marge : une valeur negative signifie une penetration.

## Resultats mesures

Dans `coal_results_all/` :

- 100 candidats testes, 12 retenus, 10 rejetes pour collision/marge, 69 echecs d'IK a la preprise et 9 a la prise.
- Aucune difference de decision avec les 100 resultats MuJoCo de reference.
- Candidat selectionne : indice 5 (sixieme candidat), score 0.69884.
- Levee reussie en simulation : 0.14496 m, maintien mesure 2.96 s, aucun avertissement du simulateur.
- `planned_lift.mp4` : video ; `filter_report.json` : detail ; `backend_comparison.json` : comparaison et revalidation du detour.

Le test `coal_obstacle_test/` ajoute volontairement une sphere de rayon 2.5 cm au trajet : trajet direct bloque, detour RRT-Connect trouve en 14 iterations, 3 configurations. Tous ses segments passent une revalidation a 0.005 rad avec **les deux moteurs**. Ce test est cinematique, sans transport physique du pot autour de l'obstacle.

## Reproduire le test d'obstacle et les verifications

```powershell
pixi run --manifest-path coal_env\pixi.toml python benchmark_panda_obstacle.py --checker coal --source coal_results_all --output coal_obstacle_test
pixi run --manifest-path coal_env\pixi.toml python -m unittest test_collision_planning.py test_coal_collision.py
pixi run --manifest-path coal_env\pixi.toml python compare_collision_backends.py
```

Les tests utilisent aussi les fichiers de reference existants dans `collision_obstacle_test/`. La comparaison utilise `collision_results_all/filter_report.json`. Neuf tests ont passe : quatre tests de planificateur, deux de geometrie MuJoCo et trois d'integration Coal (distance/marge analytique, accord sur 101 configurations Panda, marge positive sur le vrai modele). Un test interdit explicitement les appels de collision MuJoCo pendant les requetes Coal.

## Limites

Les memes limites du pipeline precedent restent applicables : verifications discretes (0.01 rad par defaut), une initialisation d'IK par etape, environnement connu, collisions convexes et exclusions du modele. Les 12 prises sont valides selon ces verifications ; seule la prise selectionnee a ete executee. La levee avec objet tenu n'est pas preplanifiee. Une marge de 2 mm en simulation n'est pas une marge validee pour le robot reel.

References : [Coal et son API Python](https://github.com/coal-library/coal), [filtrage des paires MuJoCo 3.3.7](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c).
