# TP Panda dans Google Colab

Ouvrir **Panda_MuJoCo_Colab.ipynb** dans [Google Colab](https://colab.research.google.com/) via **Fichier → Importer un notebook**, puis **Exécution → Tout exécuter**. Choisir une session CPU neuve. Le notebook est autonome : aucun autre fichier de ce dossier n’est à importer.

Il installe MuJoCo 3.3.7, télécharge le Panda MuJoCo Menagerie et le pot French’s YCB, crée une table, affiche une vidéo, et fournit un contrôleur Python de pose par cinématique inverse, fermeture par impédance, soulèvement et maintien en hauteur. Les téléchargements utilisent des commits figés ; les licences des modèles sont conservées dans `assets/`.

Le TP du groupe 2 utilisant votre robot NERO bimanuel est maintenant disponible dans `NERO_Bimanuel_Colab.ipynb`, avec l’archive `NERO_Colab_assets.zip` et le guide `README_NERO.md`.

## Repère de la pince

Le fichier Hydrax fourni contient un site `gripper` sous le corps `hand`, décalé de `[0, 0, 0.1]` m, sans rotation locale. Dans notre scène, le site équivalent est nommé `grasp_center`. Son +z pointe vers les doigts et son axe y est l’axe d’ouverture. `lab.transform()` renvoie `T_world_tcp`, en mètres. La scène utilise le modèle Menagerie, pas les réglages de collision propres à Hydrax.

## Commande

- Bras : IK 6D par Jacobien amorti, limites articulaires, interpolation quintique puis commande en couple PD avec compensation des efforts de biais. L’IK n’écrit pas dans l’état du robot simulé.
- Doigts : `F = clip(K*(q_des-q) - D*qdot, -Fmax, Fmax)`, moteurs directs de force. K = 400 N/m, D = 8 N·s/m, Fmax = 12 N par doigt. Les gains sont réglables.
- Les contacts restent actifs. La cible de prise est volontairement haute pour dégager le bouchon.

## Validation réalisée localement

MuJoCo 3.3.7, Python 3.12 sous Windows : compilation de la scène, IK de pose, détection d’une cible hors de portée, vérification du repère, suivi dynamique, contact bilatéral, saturation des commandes et absence d’avertissement numérique. La vidéo a également été générée et inspectée.

La séquence commande une montée de 15 cm en 4 s, puis maintient la pince fermée pendant 3 s. Essai de référence sans rendu : élévation réelle du pot ≈ 13,94 cm ; garde finale du bas du pot au-dessus de la table ≈ 13,69 cm ; contact bilatéral pendant tout le maintien ; dérive relative pendant le maintien ≈ 0,013 mm. Le PD sous charge présente une erreur finale du TCP d’environ 7,7 mm. Ces résultats concernent la trajectoire et le modèle testés ; les valeurs peuvent varier légèrement avec le rendu.

`lift_success` vérifie une élévation réelle d’au moins 10 cm, une garde à la table d’au moins 8 cm pendant le maintien, aucun contact pot–table, des contacts bilatéraux sur au moins 99 % des échantillons et moins de 1 cm de dérive relative pot/pince pendant le maintien. Le pot garde sa masse de 0,603 kg et son articulation libre. Aucun collage ou déplacement direct de l’objet n’est utilisé.

Le solveur utilise `cone="elliptic"`, `impratio="10"` et `noslip_iterations="20"` pour réduire la dérive tangentielle des contacts souples. Il s’agit de réglages numériques de simulation, sans calibration des matériaux réels. [Documentation MuJoCo sur les contacts et le glissement](https://mujoco.readthedocs.io/en/3.2.5/modeling.html).

**L’exécution dans un compte Google Colab n’a pas été effectuée ici.** Le notebook configure OSMesa avant d’importer MuJoCo, pour un rendu sans fenêtre sur CPU. La validation locale utilise le backend Windows. Les gains, les contacts et la géométrie sont pédagogiques ; l’IK n’effectue pas de planification anticollision. Le maintien est validé sur 3 s, sans perturbation extérieure. Contact-GraspNet n’est pas intégré à ce stade.

## Fichiers

- `Panda_MuJoCo_Colab.ipynb` : notebook autonome à importer dans Colab.
- `panda_lab.py` : scène, IK, contrôleur, rendu.
- `lift_metrics.py` : mesures de l’élévation réelle et validation du maintien.
- `fetch_assets.py` : téléchargement ciblé des modèles.
- `test_panda_lab.py` : vérifications physiques et numériques.
- `panda_demo.mp4` : vidéo de la simulation testée.
- `validation.json` : mesures de l’essai local.
- `make_notebook.py` : régénération du notebook après modification des scripts.

Pour une exécution locale avec Python disponible :

```bash
pip install -r requirements.txt
python fetch_assets.py
python test_panda_lab.py
python panda_lab.py --video
```

## Sources

- [Panda Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_emika_panda)
- [French’s / YCB mustard bottle](https://github.com/elpis-lab/YCB_Dataset/blob/9e8c6488a2ff673d9aa48a91492fb89423c1b106/ycb/mustard_bottle.xml)
- [Panda Hydrax fourni](https://github.com/MaximeSabbah/hydrax/blob/feedback-mppi-panda/hydrax/models/panda/panda.xml)
- [MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html)
- [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet)
