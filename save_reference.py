#!/usr/bin/env python3
import requests, os, yaml, cv2, numpy as np, sys
cfg_path = os.path.expanduser("~/watermeter/config.yaml")
if len(sys.argv)>1: cfg_path=sys.argv[1]
cfg=yaml.safe_load(open(cfg_path,"r"))
url=cfg["esp32"]["base_url"].rstrip("/")+"/capture_with_flashlight"
ref=os.path.expanduser(cfg["alignment"]["reference_path"])
print(f"Capturing from {url} ...")
r=requests.get(url, timeout=8); r.raise_for_status()
img=cv2.imdecode(np.frombuffer(r.content,dtype=np.uint8), cv2.IMREAD_COLOR)
os.makedirs(os.path.dirname(ref), exist_ok=True)
cv2.imwrite(ref, img)
print(f"Saved reference to {ref}")
