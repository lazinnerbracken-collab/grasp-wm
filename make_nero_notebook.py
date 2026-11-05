"""Package the supplied NERO meshes and build its self-contained Colab notebook."""
from pathlib import Path
import hashlib
import zipfile
import nbformat as nbf

archive = Path('NERO_Colab_assets.zip')
paths = [Path('nero_assets/nero_dual_arm.mpc.xml'), Path('nero_assets/LICENSE.txt'),
         Path('nero_assets/MPC_MODEL_INTEGRATION.md')]
paths += sorted(Path('nero_assets/meshes').glob('*.stl'))
paths += sorted(p for p in Path('assets/mustard').rglob('*') if p.is_file())
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        z.write(p, p.as_posix())
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# TP NERO bimanuel — MuJoCo / Google Colab

Ce notebook utilise **votre modèle NERO**, issu du dossier `models/models/agx_arm_urdf/nero_dual_arm`.
Le modèle source choisi est **nero_dual_arm.mpc.xml** : deux bras de 7 articulations et deux
glissières par pince, soit **18 actionneurs**. Le robot reste fixé au sol.

1. Importer ce notebook dans [Google Colab](https://colab.research.google.com/).
2. Utiliser une session **CPU neuve**, puis **Exécution → Tout exécuter**.
3. À la cellule d’import, sélectionner **NERO_Colab_assets.zip**, fourni avec ce notebook.

L’archive contient les maillages NERO de votre dossier et le pot French’s YCB. Le notebook
installe MuJoCo, génère une table et le pot, affiche une vidéo, résout l’IK de pose de chaque bras,
puis ferme une pince par impédance, soulève le pot et le maintient trois secondes en hauteur.
L’autre bras est commandé en position d’attente.
`ARM = 'right'` ou `'left'` choisit le bras qui saisit le pot ; **les deux bras sont présents et commandés**.
Ce TP ne réalise pas un passage du pot entre les bras ni une saisie simultanée à deux pinces.

Tests effectués localement sous Windows avec MuJoCo 3.3.7. L’exécution sur un compte Colab
n’a pas été effectuée ici. La session Colab utilise OSMesa pour le rendu CPU sans fenêtre.
''')
code('''import subprocess, sys, os
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'libosmesa6'], check=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'mujoco==3.3.7', 'numpy>=1.26,<3', 'scipy>=1.11,<2',
    'imageio>=2.34,<3', 'imageio-ffmpeg==0.6.0', 'matplotlib>=3.7'])
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
print('MuJoCo installé. Rendu CPU OSMesa configuré.')''')
md('''## 1. Importer les fichiers de votre robot

Le modèle ne dépend d’aucun chemin Windows dans Colab. Les fichiers sont extraits localement
dans la session. Le code vérifie l’intégrité de l’archive attendue avant extraction.
Les licences du robot et du pot sont incluses. Votre dossier original n’a pas été modifié.
''')
code(f'''from pathlib import Path
import zipfile, hashlib
archive = Path('NERO_Colab_assets.zip')
if not archive.exists():
    from google.colab import files
    uploaded = files.upload()  # Choisir NERO_Colab_assets.zip
    if archive.name not in uploaded:
        raise FileNotFoundError('Sélectionner le fichier NERO_Colab_assets.zip fourni.')
expected_sha256 = '{digest}'
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha256:
    raise ValueError('Archive différente : utiliser celle fournie avec ce notebook.')
destination = Path.cwd().resolve()
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        target = (destination / info.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Chemin non autorisé dans l’archive.')
    z.extractall(destination)
print('Modèles NERO et French’s disponibles.')''')
md(r'''## 2. Repères des pinces et conventions

Le fichier MuJoCo fourni n’a pas de site TCP. Nous ajoutons :

| Bras | Site de commande | Corps parent |
|---|---|---|
| Droit | `right_grasp_center` | `right_link7` |
| Gauche | `left_grasp_center` | `left_link7` |

Dans le repère `link7`, chaque site a pour position approximative **[0,175 ; 0 ; −0,0235] m**.
C’est l’origine commune des deux glissières à fermeture nulle. Les axes sont ceux de `gripper_base`,
avec le quaternion MuJoCo **wxyz ≈ [0,5 ; −0,5 ; 0,5 ; −0,5]**.
Ces valeurs sont tirées des transformations du modèle fourni, sans reprendre le décalage du Panda.
Le point est un **TCP nominal**, pas un centre de contact mesuré ; sa calibration physique reste à faire.

+z du TCP pointe vers les doigts, y est l’axe d’ouverture. Les axes visibles sont x rouge,
y vert, z bleu. `lab.transform(side)` donne **T_world_tcp**.
Les orientations cibles sont des matrices 3 × 3 exprimées dans le monde, les positions en mètres.

**Attention au signe des doigts NERO :** fermé `(0, 0)` ; ouvert `(+0,05, −0,05)` m.
L’ouverture nominale totale est donc **q₁ − q₂**, entre 0 et 0,10 m.
Pour une ouverture désirée w, les consignes sont `(w/2, −w/2)`.

Chaque moteur de pince applique :

$$F_i = operatorname{clip}left(K(q_{i,d}-q_i)-Ddot q_i,;-F_{max},;F_{max}ight).$$

K = 300 N/m, D = 5 N·s/m, Fmax = 8 N par doigt (limite du modèle : 10 N).
Le bras reçoit un PD articulaire et la compensation des efforts de biais, limités à ±16 N·m
comme dans le fichier source. Ces valeurs décrivent le modèle de recherche fourni.
''')
md('''## 3. Adaptation du modèle et contrôleur Python

Les collisions du modèle source sont désactivées. Pour ce TP, les copies de géométrie
destinées aux collisions sont activées. MuJoCo utilise leurs **enveloppes convexes** : elles
restent une approximation de la géométrie réelle, surtout pour les formes concaves.
Deux paires de carters voisins par bras sont exclues (`root`/`link1`, `link5`/`link7`), car leurs
maillages se chevauchent dans l’assemblage fourni. Les contacts robot–table, robot–pot
et entre les bras restent actifs. Les contacts non attendus dépassant 1 mm sont signalés.

Le pas est 1 ms, le solveur Newton, l’intégrateur `implicitfast`. Pour réduire la dérive
tangentielle des contacts souples : `cone='elliptic'`, `impratio=10`, `noslip_iterations=20`.
Ces réglages numériques ne sont pas une calibration physique des matériaux ; voir la
[documentation MuJoCo](https://mujoco.readthedocs.io/en/3.2.5/modeling.html).
Les capteurs bruités inutilisés
du fichier source sont retirés. Les limites, inerties et échelles des maillages sont conservées.

L’IK utilise une optimisation bornée avec plusieurs initialisations ; elle ne modifie pas
l’état du robot simulé. Les consignes sont ensuite interpolées et suivies par la dynamique.
Elle ne planifie pas l’évitement des obstacles. Les tests de contacts ne remplacent pas
un planificateur anticollision, surtout si vous modifiez les cibles.
''')
code('%%writefile lift_metrics.py\n' + Path('lift_metrics.py').read_text())
code('%%writefile nero_lab.py\n' + Path('nero_lab.py').read_text())
code('''from nero_lab import build_scene, NeroLab, Recorder, DOWN
import numpy as np
import mujoco
ARM = 'right'  # 'right' ou 'left' ; relancer depuis cette cellule après modification.
scene_path = build_scene(side=ARM)
lab = NeroLab(scene_path)
print('MuJoCo', mujoco.__version__, '| actionneurs :', lab.model.nu)
for side in ['right', 'left']:
    print(side, 'T_world_tcp :\\n', lab.transform(side))''')
md('''## 4. Approche, fermeture, soulèvement et maintien

La table est à z = 0,40 m, le pot est devant le bras sélectionné. Les deux bras atteignent
d’abord une préprise et une position d’attente. Le bras sélectionné descend, ferme les doigts
sur le pot, puis monte vers une cible 15 cm plus haute pendant 4 s. La pince reste fermée
pendant les 3 s de maintien en hauteur. Le pot reste libre et conserve sa masse de 0,603 kg :
il n’est pas soudé à la pince. Le PD chargé a une petite erreur de suivi ; le pot monte d’environ 14 cm.

`lift_success` exige au moins 10 cm d’élévation réelle du pot, une garde à la table de 8 cm
pendant le maintien, aucun contact avec la table, des contacts bilatéraux sur au moins 99 %
des échantillons et une dérive relative pot/pince inférieure à 1 cm.
''')
code('''from IPython.display import Video, display
import json
recorder = Recorder(lab.model, 'nero_demo.mp4')
try:
    results = lab.demo(side=ARM, capture=recorder)
finally:
    recorder.close()
print(json.dumps(results, indent=2))
display(Video('nero_demo.mp4', embed=True, width=720))
assert results['bilateral_contact'], 'La pince ne touche pas le pot des deux côtés.'
assert results['lift_success'], 'Soulèvement ou maintien échoué : consulter les mesures.'
assert not any(results['warnings']), 'Avertissement numérique MuJoCo.'
assert not results['unexpected_contacts_over_1mm'], 'Inspecter les contacts supplémentaires.' ''')
md('''## 5. Mesures d’ouverture, commandes et forces de contact

Une commande moteur limitée n’impose pas la même borne aux pics de force de contact.
La simulation mesure donc séparément la force normale entre chaque doigt et le pot.
Le déplacement réel du pot et sa garde à la table sont également mesurés. Le maintien
est testé pendant 3 s, sans perturbation extérieure.
''')
code('''import matplotlib.pyplot as plt
t = np.array([row['time'] for row in lab.log])
opening = np.array([row[ARM]['opening'] for row in lab.log])
force = np.array([row[ARM]['force'] for row in lab.log])
contact = np.array([row[ARM]['contact'] for row in lab.log])
clearance = np.array([row[ARM]['clearance_m'] for row in lab.log])
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t, 1000*opening); axs[0].set_ylabel('Ouverture [mm]')
axs[1].plot(t, force[:,0], label='Doigt 1')
axs[1].plot(t, force[:,1], '--', label='Doigt 2')
axs[1].set_ylabel('Commande [N]'); axs[1].legend()
axs[2].plot(t, contact[:,0], label='Contact doigt 1')
axs[2].plot(t, contact[:,1], '--', label='Contact doigt 2')
axs[2].set_ylabel('Normale [N]'); axs[2].legend()
axs[3].plot(t, 100*clearance, label='Bas du pot au-dessus du plateau')
axs[3].axhline(0, color='black', linewidth=.8)
axs[3].set_ylabel('Garde [cm]'); axs[3].legend()
axs[3].set_xlabel('Temps [s]')
for ax in axs: ax.grid(alpha=.3)
plt.tight_layout(); plt.show()''')
md('''## 6. Commander des poses indépendantes aux deux bras

Exercice facultatif. `poses` peut contenir un seul bras ou les deux. Pour un seul bras, l’autre
conserve sa consigne. Modifier la position et l’orientation souhaitées dans le repère monde.
La cellule réinitialise la scène, puis garde les pinces ouvertes pendant le mouvement.
''')
code('''RUN_CUSTOM = False
poses = {
    'right': (np.array([-.25, -.45, .76]), DOWN.copy()),
    'left': (np.array([.45, -.30, .80]), DOWN.copy()),
}
if RUN_CUSTOM:
    lab.reset()
    recorder = Recorder(lab.model, 'nero_custom.mp4')
    try:
        print(lab.reach_poses(poses, duration=4., capture=recorder))
    finally:
        recorder.close()
    print('Contacts supplémentaires :', sorted(lab.unexpected_contacts))
    display(Video('nero_custom.mp4', embed=True, width=720))

# Gains à modifier avant de relancer la démonstration :
# lab.k_finger = 200.       # N/m
# lab.d_finger = 8.         # N.s/m
# lab.force_limit = 6.      # N par doigt, >0 et <=10
# Une fermeture isolée à la pose actuelle :
# lab.segment(openings={'right': 0.}, duration=1.5)''')
md('''## Validation et limites

Les deux variantes ont été testées localement : élévation du pot d’environ 14 cm, garde finale
d’environ 13,7 cm au-dessus de la table et contact bilatéral conservé pendant 3 s.
Aucun avertissement numérique ni contact inattendu de plus de 1 mm détecté sur ces essais.
Les nombres peuvent varier légèrement selon le rendu et la plateforme.

L’IK d’une cible hors de portée est rejetée. Le notebook ne garantit pas l’absence de collision
pour d’autres poses. La manipulation coordonnée d’un même objet et le transfert
entre pinces restent à développer et tester. Contact-GraspNet n’est pas exécuté ici.

Pour utiliser ensuite ses poses : `T_world_tcp = T_world_camera @ T_camera_grasp @ T_grasp_tcp`.
Identifier les conventions de la sortie du réseau et calibrer la dernière transformation pour **NERO**.

Si OpenGL échoue dans Colab, redémarrer la session puis tout exécuter dans l’ordre, afin que
OSMesa soit configuré avant le premier import MuJoCo. Conserver l’archive avec le notebook.

Sources : votre fichier `nero_dual_arm.mpc.xml` et ses maillages ; la documentation fournie
indique une origine `BhavikShangari/nero-touch` au commit `c6199860a9e9a5c50fc2c93750809d96da4c6819`.
Le [pot French’s YCB](https://github.com/elpis-lab/YCB_Dataset/blob/9e8c6488a2ff673d9aa48a91492fb89423c1b106/ycb/mustard_bottle.xml)
est le même que dans le TP Panda.
''')
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
               'language_info': {'name':'python'}, 'colab': {'name':'NERO_Bimanuel_Colab.ipynb'}}
nbf.validate(nb)
for i,c in enumerate(cells):
    if c.cell_type == 'code':
        compile(c.source.split('\n',1)[1] if c.source.startswith('%%writefile') else c.source, f'cell {i}', 'exec')
nbf.write(nb, 'NERO_Bimanuel_Colab.ipynb')
print('Notebook validated; assets:', round(archive.stat().st_size/1e6,2), 'MB; SHA256:', digest)
