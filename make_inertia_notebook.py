from pathlib import Path
import zipfile
import nbformat as nbf

with zipfile.ZipFile('YCB_Inertia_Inputs.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(Path('ycb_inertia_inputs').glob('*')):
        z.write(p,p.as_posix())
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))
md('''# YCB : estimer centre de masse et inertie avec Embodied-R1.5

Expérience initiale sur **006_mustard_bottle / French’s**. Les entrées sont trois vues
du maillage texturé, une masse connue, les dimensions métriques et un repère objet explicite.
La sortie demandée est le centre de masse et le tenseur d’inertie complet au centre de masse.

**Embodied-R1.5 n’est pas présenté par ses auteurs comme un estimateur d’inertie validé.**
Nous testons ici une utilisation expérimentale de ses capacités image+texte. La masse seule
et les images extérieures ne rendent pas la distribution interne de masse identifiable.
Une bouteille pleine, partiellement remplie ou lestée peut avoir des propriétés différentes.
Les nombres du VLM sont des estimations conditionnelles, pas des mesures.

Dans Colab : choisir **un environnement GPU**, importer ce notebook et exécuter les cellules.
Importer **YCB_Inertia_Inputs.zip** quand demandé. Les modèles sont volumineux et nécessitent
un téléchargement Hugging Face. Le mode 4 bits réduit la mémoire, sans garantie de tenir
sur tous les GPU Colab ; il peut également modifier les résultats par rapport à BF16.

**Statut :** vues et calculs géométriques préparés localement, parseur et contrôles testés.
Les poids Embodied-R1.5 n’ont pas été chargés ici : aucune prédiction VLM n’a été obtenue
et l’inférence GPU/Colab reste à exécuter.
''')
code('''%pip -q install "transformers>=4.57,<5" "accelerate>=1.0" "bitsandbytes>=0.48" pillow numpy
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Activer un GPU dans Exécution → Modifier le type d’exécution.')
print(torch.cuda.get_device_name(0))
print('Mémoire GPU totale [GiB] :', torch.cuda.get_device_properties(0).total_memory / 2**30)''')
md('''## 1. Entrées et masse connue

Les vues sont des **rendus synthétiques**, pas des observations du contenu interne.
La masse initiale de 0,603 kg vient du XML YCB dérivé utilisé dans les TP précédents :
remplacer cette valeur par votre mesure si elle diffère. Les images ne doivent pas changer
de repère indépendamment des coordonnées du maillage.
''')
code('''from pathlib import Path
import zipfile, json
archive=Path('YCB_Inertia_Inputs.zip')
if not archive.exists():
    from google.colab import files
    files.upload()
root=Path.cwd().resolve()
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        if not (root/member.filename).resolve().is_relative_to(root):
            raise ValueError('Chemin invalide dans l’archive.')
    z.extractall(root)
case=json.loads(Path('ycb_inertia_inputs/case.json').read_text())
KNOWN_MASS_KG=0.603  # Remplacer par la masse mesurée de VOTRE exemplaire.
if KNOWN_MASS_KG <= 0:
    raise ValueError('La masse doit être positive.')
case['mass_kg']=KNOWN_MASS_KG
case['mass_provenance']='Input KNOWN_MASS_KG; default 0.603 kg from the YCB-derived XML.'
print(json.dumps(case,indent=2))''')
code('%%writefile ycb_inertia.py\n'+Path('ycb_inertia.py').read_text())
code('''from PIL import Image
from IPython.display import display
from ycb_inertia import make_prompt, box_baseline, parse_answer, validate_estimate, compare_to_baseline
images=[Image.open(f'ycb_inertia_inputs/view_{i}.png').convert('RGB') for i in [1,2,3]]
for image in images: display(image)
prompt=make_prompt(case)
print(prompt)''')
md('''## 2. Inférence avec le modèle demandé

Modèle : `IffYuan/Embodied-R1.5`, construit sur Qwen3-VL-8B-Instruct. Le format des entrées
suit l’exemple Hugging Face officiel du dépôt ; le JSON de propriétés physiques est notre
protocole expérimental, pas une sortie native garantie.

Le tenseur demandé est **au centre de masse**, dans des axes parallèles au repère du maillage.
L’origine du maillage n’est pas supposée égale au centre de masse. Les valeurs sont en
mètres, kilogrammes et kg·m², jamais en coordonnées image normalisées.
''')
code('''from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
import transformers, time
MODEL_ID='IffYuan/Embodied-R1.5'
REVISION='main'  # Pour reproduire une expérience, remplacer par son commit Hugging Face.
USE_4BIT=True
dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
kwargs=dict(device_map='auto', dtype=dtype, attn_implementation='sdpa', revision=REVISION)
if USE_4BIT:
    kwargs['quantization_config']=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True)
processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION)
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,**kwargs)
model.eval()
print('Modèle chargé ; quantification 4 bits :',USE_4BIT)''')
code('''messages=[{'role':'user','content':
    [{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=processor(text=[text],images=images,return_tensors='pt').to(model.device)
inputs.pop('token_type_ids',None)
start=time.time()
with torch.inference_mode():
    generated=model.generate(**inputs,max_new_tokens=2048,do_sample=False)
new_tokens=generated[:,inputs.input_ids.shape[1]:]
raw=processor.batch_decode(new_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
Path('vlm_raw_response.txt').write_text(raw,encoding='utf-8')
run=dict(model=MODEL_ID,requested_revision=REVISION,
         resolved_revision=getattr(model.config,'_commit_hash',None),
         transformers_version=transformers.__version__,torch_version=torch.__version__,
         quantized_4bit=USE_4BIT,elapsed_seconds=time.time()-start,
         input_case=case,prompt=prompt)
Path('vlm_run_metadata.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
print(raw)''')
md('''## 3. Contrôles physiques et comparaison

Contrôles nécessaires : masse inchangée, repère explicite, centre de masse dans la boîte
englobante, matrice symétrique définie positive, inégalités triangulaires des moments
principaux et bornes liées à la taille de l’objet. **Passer ces contrôles ne prouve pas l’exactitude.**
La sortie brute est conservée même si elle est invalide ; aucun nombre n’est corrigé silencieusement.

La référence simple est un parallélépipède plein homogène ayant la masse et les dimensions
extérieures du pot : ce modèle est volontairement grossier, **ce n’est pas une vérité terrain**.
Pour une étude plus poussée, ajouter un maillage fermé de densité uniforme, puis des scénarios
de coque+contenu avec différents niveaux de remplissage.
''')
code('''estimate=parse_answer(raw)
report=validate_estimate(estimate,case)
Path('vlm_estimate.json').write_text(json.dumps(estimate,indent=2),encoding='utf-8')
Path('physical_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Contrôles :',json.dumps(report,indent=2))
baseline=box_baseline(case['mass_kg'],case['bounds_m'])
print('Référence homogène, pas vérité terrain :',json.dumps(baseline,indent=2))
if report['valid']:
    comparison=compare_to_baseline(estimate,baseline)
    print(json.dumps(comparison,indent=2))
else:
    print('Ne pas utiliser cette sortie dans MuJoCo sans résoudre les problèmes indiqués.')''')
md('''## 4. Fragment MuJoCo à examiner, sans modification automatique des scènes

Si les contrôles passent, on peut produire un fragment inertiel pour le corps objet
dont le repère est précisément celui de ce maillage. Les coefficients `fullinertia` suivent
l’ordre Ixx, Iyy, Izz, Ixy, Ixz, Iyz. Remplacer l’ancien élément `inertial`, sans le dupliquer.
Un changement de repère exige la transformation du centre de masse et du tenseur.
''')
code('''import numpy as np
import xml.etree.ElementTree as ET
if report['valid']:
    tensor=np.array(estimate['inertia_com_kg_m2'])
    components=[tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]]
    inertial=ET.Element('inertial',mass=str(case['mass_kg']),
        pos=' '.join(map(str,estimate['com_m'])),
        fullinertia=' '.join(map(str,components)))
    fragment=ET.tostring(inertial,encoding='unicode')
    print(fragment)
    Path('estimated_inertial_fragment.xml').write_text(fragment)
    print('Fragment estimé à examiner. Les notebooks Panda et NERO ne sont pas modifiés.')''')
md('''## Protocole d’évaluation sur YCB

Pour chaque objet, fournir plusieurs vues, une masse de provenance vérifiée, l’échelle
métrique et le repère. Enregistrer la révision exacte du modèle, la quantification, le prompt,
les vues et la réponse brute. Tester la sensibilité aux vues, à l’identification de l’objet et
aux hypothèses sur son remplissage. Une réponse « confiance élevée » n’est pas une incertitude calibrée.

Comparer VLM et références géométriques. Sans mesures physiques indépendantes, présenter
des **écarts entre modèles**, pas une erreur par rapport à une vérité terrain.
Pour une vraie validation, utiliser une mesure indépendante du centre de masse et une identification
dynamique des inerties avec mouvements et forces/couples connus. Le succès d’une prise MuJoCo
ne valide pas à lui seul les propriétés inertiales réelles.

Attention : notre XML actuel fournit `pos="0 0 0"` et `diaginertia="0.001 0.001 0.001"`.
Ces paramètres de simulation ne constituent pas des mesures de référence du pot.
Pour un liquide ou un contenu mobile, une inertie constante de corps rigide est une approximation.

Sources : [Embodied-R1.5](https://github.com/pickxiguapi/Embodied-R1.5),
[exemple d’inférence](https://github.com/pickxiguapi/Embodied-R1.5/blob/main/inference/hf_example.py),
[poids](https://huggingface.co/IffYuan/Embodied-R1.5),
[quantification](https://huggingface.co/docs/transformers/quantization/bitsandbytes),
[maillage YCB dérivé](https://github.com/elpis-lab/YCB_Dataset).
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'YCB_Embodied_R15_Inertia_Colab.ipynb'}})
nbf.validate(nb)
for i,c in enumerate(cells):
    if c.cell_type=='code':
        src=c.source.split('\n',1)[1] if c.source.startswith('%%writefile') else '\n'.join(l for l in c.source.splitlines() if not l.startswith('%pip'))
        compile(src,f'cell_{i}','exec')
nbf.write(nb,'YCB_Embodied_R15_Inertia_Colab.ipynb')
print('Notebook generated; Python cells compile. GPU inference NOT executed.')
