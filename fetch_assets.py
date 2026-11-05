"""Download only the Panda and French's YCB assets used by this exercise."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PANDA_REV = "c96a32d28fb5da84da38c1da4d749e7a13212855"
YCB_REV = "9e8c6488a2ff673d9aa48a91492fb89423c1b106"

def download(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    response = session.get(url, timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination

def fetch_assets(root="assets"):
    root = Path(root).resolve()
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{PANDA_REV}/franka_emika_panda/"
    xml = download(base + "panda.xml", root / "panda/panda.xml")
    meshes = {node.get("file") for node in ET.parse(xml).findall(".//mesh") if node.get("file")}
    jobs = [(base + "assets/" + name, root / "panda/assets" / name) for name in meshes]
    jobs += [(base + "LICENSE", root / "panda/LICENSE")]
    ycb = f"https://raw.githubusercontent.com/elpis-lab/YCB_Dataset/{YCB_REV}/"
    jobs += [(ycb + "ycb/mustard_bottle.xml", root / "mustard/source.xml")]
    jobs += [(ycb + "LICENSE", root / "mustard/LICENSE")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda pair: download(*pair), jobs))
    # Follow the source XML so paths do not depend on guesses about the dataset layout.
    source = ET.parse(root / "mustard/source.xml")
    jobs = []
    for node in source.findall(".//mesh") + source.findall(".//texture"):
        name = node.get("file")
        if name:
            jobs.append((ycb + "ycb/" + name, root / "mustard" / name))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda pair: download(*pair), jobs))
    print("Assets ready:", root)
    return root

if __name__ == "__main__":
    fetch_assets()
