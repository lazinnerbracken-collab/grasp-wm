"""Build the standalone notebook: no local helper uploads required in Colab."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# TP Panda — MuJoCo dans Google Colab

**Groupe 1 : un Panda, une table, le pot French’s (YCB mustard bottle).**

Ce notebook autonome installe MuJoCo, télécharge les modèles, crée la scène, résout une
cinématique inverse de pose **position + orientation**, puis ferme la pince avec une
**loi d’impédance en force**, soulève le pot et le maintient trois secondes en hauteur.
La simulation s’affiche sous forme de vidéo intégrée.

Dans Colab : **Fichier → Importer un notebook**, sélectionner ce fichier, puis
**Exécution → Tout exécuter**. Un environnement **CPU** suffit : le rendu OSMesa est logiciel.
Exécuter dans une session neuve pour configurer OpenGL avant le premier import de MuJoCo.
Les téléchargements demandent une connexion Internet. Aucun compte ou jeton n’est nécessaire.

La trajectoire de démonstration utilise une pose connue du pot. Contact-GraspNet n’est pas exécuté ici.
La cinématique inverse n’est pas un planificateur avec évitement d’obstacles : si vous changez
la cible, vérifiez la trajectoire et les collisions. Le TP mesure le déplacement du pot,
sa séparation de la table et son glissement pendant le maintien en hauteur.
""")
code("""# Installation dans la machine Colab (pas sur votre ordinateur).
import subprocess, sys, os
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'libosmesa6'], check=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'mujoco==3.3.7', 'numpy>=1.26,<3', 'scipy>=1.11,<2', 'requests>=2.31,<3',
    'imageio>=2.34,<3', 'imageio-ffmpeg==0.6.0', 'pillow>=10', 'matplotlib>=3.7'])
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
print('Installation terminée. Backend : OSMesa (CPU).')""")
md("""## 1. Modèles publics et téléchargements

- [Panda MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), version figée dans le code.
- [Pot French’s YCB, modèle mustard_bottle](https://github.com/elpis-lab/YCB_Dataset/blob/main/ycb/mustard_bottle.xml), maillage texturé et trois maillages de collision convexes. Dimensions conservées (environ 5,8 × 9,6 × 19,1 cm), masse du fichier source : 0,603 kg.
- [Votre Panda Hydrax](https://github.com/MaximeSabbah/hydrax/blob/feedback-mppi-panda/hydrax/models/panda/panda.xml) : son site `gripper` est attaché à `hand` à `pos="0 0 0.1"`. Nous reprenons cet offset pour `grasp_center`.

La scène utilise **Menagerie**, avec ses collisions actives, et remplace les actionneurs
de position par des moteurs de couple/force. Les licences des deux modèles sont téléchargées
avec les fichiers. Les inerties et les contacts restent des approximations de simulation.
""")
code("%%writefile fetch_assets.py\n" + Path("fetch_assets.py").read_text())
code("from fetch_assets import fetch_assets\nasset_root = fetch_assets()")
md(r"""## 2. Scène et contrôleur Python

Le code ci-dessous est enregistré dans `panda_lab.py` : vous pouvez le modifier directement.
La table a son plateau à z = 0,4 m. Le robot est fixé sur le plateau ; le pot est libre
et soumis à la gravité et aux contacts.

**Repères et unités** : mètres, radians, secondes, newtons, newtons-mètres.
`hand` est l’origine de la main ; `grasp_center` est notre TCP, au centre nominal entre les doigts.
Son orientation est celle de `hand`. +z pointe de la main vers les doigts ; l’axe y est
l’axe d’ouverture. Les axes affichés sont x rouge, y vert, z bleu.

Le TCP est fixe dans la main : ce n’est pas un point de contact mesuré et il ne suit pas
une éventuelle asymétrie des doigts. La matrice `T_world_tcp` transforme des coordonnées
du TCP vers le monde. Son inverse transforme du monde vers le TCP.

Pour les doigts, avec une consigne par doigt q_d en mètres :

$$F_i = \mathrm{clip}\left(K(q_d-q_i)-D\dot q_i,\;-F_{max},\;F_{max}\right).$$

Valeurs initiales : K = 400 N/m, D = 8 N·s/m, Fmax = 12 N **par doigt**.
q_d = 0,04 ouvre ; q_d = 0 ferme. L’ouverture nominale totale est q₁ + q₂.
La consigne diminue progressivement et le contact arrête les doigts avant q = 0.
Il s’agit d’une impédance articulaire des doigts ; la saturation de commande ne garantit
pas une borne identique sur les forces transitoires de contact, qui sont mesurées séparément.

Le bras reçoit un PD articulaire avec compensation des efforts de biais MuJoCo.
L’IK utilise le Jacobien du TCP et une pseudo-inverse amortie ; elle ne modifie pas l’état
du robot simulé. Les mouvements suivants sont intégrés par la dynamique MuJoCo.
""")
md("""**Contacts pour le soulèvement.** Le solveur utilise `cone='elliptic'`, `impratio=10`
et `noslip_iterations=20` pour réduire la dérive tangentielle des contacts souples.
Ce sont des réglages numériques de simulation, pas une calibration expérimentale des matériaux.
Voir la [documentation MuJoCo sur le glissement](https://mujoco.readthedocs.io/en/3.2.5/modeling.html).
Le pot conserve son articulation libre et sa masse de 0,603 kg ; aucune liaison rigide ne le fixe à la pince.
""")
code("%%writefile lift_metrics.py\n" + Path("lift_metrics.py").read_text())
code("%%writefile panda_lab.py\n" + Path("panda_lab.py").read_text())
code("""from panda_lab import build_scene, PandaLab, Recorder, DOWN, TABLE_Z, OBJECT_XY
import numpy as np
import mujoco
print('MuJoCo', mujoco.__version__)
scene_path = build_scene(asset_root)
lab = PandaLab(scene_path)
print('Repère de pince : grasp_center, parent : hand, offset : [0, 0, 0.1] m')
print('T_world_tcp au repos :\\n', lab.transform())
print('Articulations :', lab.model.nq, 'coordonnées ;', lab.model.nu, 'moteurs')""")
md("""## 3. Démonstration : approche → fermeture → soulèvement → maintien

La cible est à 16 cm au-dessus du plateau pour éviter que la paume heurte le bouchon.
La préprise est 20 cm au-dessus. L’axe +z du TCP est orienté vers le bas, et son axe y
selon l’axe x du monde, le côté étroit du pot.

Une erreur de suivi trop grande interrompt la démonstration. Les résultats signalent
explicitement si les deux doigts sont en contact. Ensuite, une cible de montée de 15 cm est
suivie pendant 4 s, pince fermée, puis maintenue pendant 3 s. Le PD chargé peut laisser
quelques millimètres d’erreur : le déplacement du pot mesuré est d’environ 14 cm.
Relancer cette cellule réinitialise la scène.

`lift_success` exige une élévation réelle du pot d’au moins 10 cm, une garde à la table
d’au moins 8 cm durant le maintien, aucun contact avec la table, des contacts bilatéraux
sur au moins 99 % des échantillons, et moins de 1 cm de dérive relative pot/pince.
""")
code("""from IPython.display import Video, display
import json
recorder = Recorder(lab.model, 'panda_demo.mp4')
try:
    results = lab.demo(capture=recorder)
finally:
    recorder.close()
print(json.dumps(results, indent=2))
display(Video('panda_demo.mp4', embed=True, width=720))
assert results['bilateral_contact'], 'Pas de contact bilatéral : inspecter la scène.'
assert results['lift_success'], 'Soulèvement ou maintien échoué : consulter les mesures.'
assert not any(results['warnings']), 'MuJoCo a signalé une instabilité.'""")
md("""## 4. Observer l’impédance et les contacts

La commande moteur et la force normale de contact sont des grandeurs distinctes.
Une ouverture finale non nulle avec deux forces de contact positives montre que les doigts
rencontrent le pot. La quatrième courbe vérifie en plus que le bas du pot reste au-dessus
de la table. Le maintien est testé pendant 3 s, sans perturbation extérieure.
""")
code("""import matplotlib.pyplot as plt
t = np.array([row['time'] for row in lab.log])
opening = np.array([row['opening'] for row in lab.log])
command = np.array([row['force'] for row in lab.log])
contact = np.array([row['contact'] for row in lab.log])
clearance = np.array([row['clearance_m'] for row in lab.log])
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axes[0].plot(t, 1000 * opening)
axes[0].set_ylabel('Ouverture [mm]')
axes[1].plot(t, command[:, 0], label='Doigt 1')
axes[1].plot(t, command[:, 1], '--', label='Doigt 2')
axes[1].set_ylabel('Commande [N]')
axes[2].plot(t, contact[:, 0], label='Contact doigt 1')
axes[2].plot(t, contact[:, 1], '--', label='Contact doigt 2')
axes[2].set_ylabel('Force normale [N]')
axes[3].plot(t, 100 * clearance, label='Bas du pot au-dessus du plateau')
axes[3].axhline(0, color='black', linewidth=.8)
axes[3].set_ylabel('Garde [cm]'); axes[3].legend()
axes[3].set_xlabel('Temps [s]')
for ax in axes:
    ax.grid(alpha=.3)
axes[1].legend(); axes[2].legend()
plt.tight_layout(); plt.show()""")
md("""## 5. Votre pose désirée

Exercice indépendant : réinitialiser puis atteindre une pose au-dessus de la table, pince ouverte.
Modifier `desired_position` et `desired_rotation`. L’orientation est une matrice 3 × 3
qui exprime les axes du TCP dans le monde. Une cible hors de portée déclenche une erreur.
Une pose atteignable ne garantit pas une trajectoire sans obstacle.
""")
code("""from scipy.spatial.transform import Rotation
RUN_CUSTOM_POSE = False  # Passer à True pour lancer cet exercice.
desired_position = np.array([0.45, 0.10, 0.80])  # monde [m]
desired_rotation = DOWN.copy()
# Exemple de rotation supplémentaire autour de z MONDE :
# desired_rotation = Rotation.from_euler('z', 20, degrees=True).as_matrix() @ DOWN
if RUN_CUSTOM_POSE:
    lab.reset()
    recorder = Recorder(lab.model, 'custom_pose.mp4')
    try:
        error = lab.reach_pose(desired_position, desired_rotation, duration=3., capture=recorder)
        print('Erreur finale :', error)
    finally:
        recorder.close()
    display(Video('custom_pose.mp4', embed=True, width=720))""")
md("""## 6. Paramètres de la fermeture et exercices

Modifier les gains puis relancer la démonstration (cellule 3) :
- `lab.k_finger = 200.` : comparer la force de contact et l’ouverture finale.
- `lab.d_finger = 12.` : observer l’amortissement.
- `lab.force_limit = 8.` : limiter la commande par doigt (rester entre 0 et 15 N, limite des moteurs XML).
- Comparer la pose de `hand` avec celle de `grasp_center` ; expliquer le décalage de 10 cm.
- Réduire la force de serrage et observer à partir de quand `lift_success` devient faux.
- Comparer la cible de montée (15 cm), l’élévation du pot et son glissement relatif.

**Lien futur avec Contact-GraspNet.** Une prédiction dans le repère caméra doit être
convertie : `T_world_tcp = T_world_camera @ T_camera_grasp @ T_grasp_tcp`.
Il faut identifier la convention exacte de la sortie du réseau et calibrer la dernière
transformation ; ne pas supposer que le repère prédit coïncide avec `hand` ou notre TCP.
Ce notebook ne charge ni poids du réseau ni nuage de points.

**Dépannage Colab.** Si OpenGL échoue après un import antérieur, redémarrer la session
et exécuter depuis le début. Si un téléchargement échoue, relancer sa cellule.
Les fichiers téléchargés sont mis en cache dans `assets/` pour la session.
La vidéo utilise un rendu logiciel ; sa génération peut prendre plus de temps que la simulation.

Sources : [API Python MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html),
[Panda Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda),
[YCB Dataset](https://github.com/elpis-lab/YCB_Dataset),
[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet).
""")
nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"name": "Panda_MuJoCo_Colab.ipynb"}}
nbf.validate(nb)
nbf.write(nb, "Panda_MuJoCo_Colab.ipynb")
print("Notebook generated and validated")
