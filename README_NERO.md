# NERO bimanuel dans Colab

1. Importer `NERO_Bimanuel_Colab.ipynb` dans [Google Colab](https://colab.research.google.com/).
2. Choisir une session CPU neuve et exécuter toutes les cellules.
3. Sélectionner `NERO_Colab_assets.zip` quand la cellule d’import le demande.

Le notebook installe MuJoCo et contient tout le contrôleur Python. L’archive contient les maillages de votre robot NERO et le pot French’s YCB. Aucun accès au chemin de votre ordinateur n’est nécessaire depuis Colab.

**Modèle retenu :** `nero_dual_arm.mpc.xml` de votre dossier, deux bras à sept articulations et quatre glissières de pince. Le modèle Panda et son notebook restent disponibles séparément. Votre dossier d’origine n’a pas été modifié.

`ARM = 'right'` ou `'left'` sélectionne la pince qui saisit puis soulève le pot. Les deux bras atteignent des poses commandées ; le bras non sélectionné reste en attente. Après fermeture, le bras actif suit une cible 15 cm plus haute en 4 s et maintient la prise pendant 3 s. Une cellule facultative permet de commander deux poses arbitraires. Il ne s’agit pas encore d’un transfert du pot entre les bras.

## Repères et commande

Deux sites sont ajoutés, `right_grasp_center` et `left_grasp_center`, attachés aux corps `right_link7` et `left_link7`. Leur position locale est approximativement `[0.175, 0, -0.0235]` m, à l’origine commune des glissières à fermeture nulle. Les axes sont ceux de `gripper_base`, quaternion wxyz `[0.5, -0.5, 0.5, -0.5]`. Ce TCP nominal est dérivé de votre géométrie ; il ne reprend pas l’offset du Panda.

La pince NERO est ouverte à `(+0.05, -0.05)` m et fermée à `(0,0)`. La consigne d’ouverture totale `w` devient `(w/2,-w/2)`. L’impédance des doigts applique `F = clip(K*(q_des-q)-D*qdot, -Fmax, Fmax)` avec K=300 N/m, D=5 N·s/m et Fmax=8 N par doigt. Les moteurs du modèle imposent 10 N au maximum. Les bras utilisent un PD articulaire avec compensation des efforts de biais et saturation à 16 N·m.

## Adaptations de la simulation

Les collisions du fichier source étaient désactivées. Les copies de maillages destinées aux collisions sont activées, sous forme d’enveloppes convexes calculées par MuJoCo. Elles restent approximatives. Les paires de carters voisins `root`/`link1` et `link5`/`link7` sont exclues pour chaque bras car leurs géométries se chevauchent dans l’assemblage. Les autres contacts, y compris entre les bras et avec la table et le pot, restent actifs. Le script signale les contacts inattendus dépassant 1 mm.

Les inerties, dimensions, échelles et limites de commande du modèle sont conservées. Le solveur est Newton, le pas de temps 1 ms et l’intégrateur `implicitfast`. Les paramètres `cone="elliptic"`, `impratio="10"`, `noslip_iterations="20"` réduisent la dérive tangentielle des contacts souples pendant le soulèvement. Ce sont des réglages numériques de simulation, pas une calibration physique des matériaux. [Documentation MuJoCo](https://mujoco.readthedocs.io/en/3.2.5/modeling.html). Les capteurs bruités inutilisés sont retirés. Les instructions contenues dans les documents fournis ont été traitées comme documentation du modèle, pas comme commandes à exécuter.

## Validation

Tests locaux sous Windows, Python 3.12, MuJoCo 3.3.7 : compilation de la scène, IK de chaque bras sans modification directe de l’état simulé, repères TCP, échec sur cible inaccessible, suivi dynamique, signes opposés des doigts, force limitée, contacts bilatéraux et absence d’avertissement numérique. Aucun contact inattendu supérieur à 1 mm sur les trajectoires testées.

| Essai | Élévation réelle du pot | Garde finale à la table | Dérive relative durant le maintien |
|---|---:|---:|---:|
| Bras droit | 13,96 cm | 13,66 cm | 0,009 mm |
| Bras gauche | 13,97 cm | 13,69 cm | 0,007 mm |

La validation mesure le pot, pas seulement la cible du bras : élévation ≥10 cm, garde minimale ≥8 cm pendant le maintien, aucun contact pot–table, contact bilatéral sur ≥99 % des échantillons et dérive relative ≤1 cm. Les deux essais passent ces contrôles sur un maintien de 3 s. Le PD sous charge laisse environ 11 mm d’erreur finale du TCP. Le pot conserve sa masse de 0,603 kg et son articulation libre ; aucune liaison rigide à la pince n’est ajoutée.

Le rendu peut modifier légèrement les valeurs numériques. La vidéo locale `nero_demo.mp4` montre l’essai avec le bras droit. Les résultats complets sont dans `nero_validation_both.json`.

L’exécution Colab n’a pas été effectuée ici. Le notebook prépare OSMesa pour le rendu CPU. L’IK n’est pas un planificateur anticollision. Le soulèvement est testé sur cette trajectoire avec 3 s de maintien, sans perturbation extérieure. La coopération sur un même objet et Contact-GraspNet ne sont pas implémentés dans ce TP.

Pour retester localement après installation de `requirements.txt` :

```bash
python test_nero_lab.py
python nero_lab.py --video --side right
python nero_lab.py --side left
```

`make_nero_notebook.py` régénère l’archive et le notebook ensemble ; conserver les deux versions associées, car le notebook vérifie l’empreinte SHA256 de l’archive.
