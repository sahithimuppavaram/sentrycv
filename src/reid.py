"""
reid.py  --  cross-camera re-identification (lightweight version).

Turns each person's saved crops into an embedding using a small ResNet18 that
ships with torchvision (nothing extra to install), then matches the same person
across cameras by cosine similarity and assigns one GLOBAL id.

This mirrors Verkada's "track objects across cameras" (People Analytics). A
specialized re-ID model like OSNet would be more accurate, but ResNet18
features are plenty to demonstrate the idea and they run with what you already
have installed.
"""
import os
import glob
import numpy as np
from collections import defaultdict

import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


class ReID:
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

        # load a pretrained ResNet18 and chop off its classifier head so it
        # outputs a 512-dim feature vector instead of class scores
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model

        # standard ImageNet preprocessing for the crops
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _embed_paths(self, crop_paths):
        """Average the embeddings of a track's crops -> one robust vector."""
        tensors = []
        for p in crop_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(self.transform(img))
            except Exception:
                continue
        if not tensors:
            return None
        batch = torch.stack(tensors)
        feats = self.model(batch).cpu().numpy()
        v = feats.mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    def build_gallery(self, crops_root):
        """crops_root/camX/idY_*.jpg -> {(cam, local_id): embedding}"""
        gallery = {}
        for cam_dir in sorted(glob.glob(os.path.join(crops_root, "cam*"))):
            cam = os.path.basename(cam_dir)
            tracks = defaultdict(list)
            for p in glob.glob(os.path.join(cam_dir, "*.jpg")):
                local_id = os.path.basename(p).split("_")[0]  # "idY"
                tracks[local_id].append(p)
            for local_id, paths in tracks.items():
                emb = self._embed_paths(paths)
                if emb is not None:
                    gallery[(cam, local_id)] = emb
        return gallery

    def assign_global_ids(self, gallery):
        """Greedy cosine matching across tracks -> one global id per track."""
        keys = list(gallery.keys())
        global_id = {}
        next_gid = 0
        for k in keys:
            best_gid, best_sim = None, self.threshold
            for other, gid in global_id.items():
                sim = float(np.dot(gallery[k], gallery[other]))
                if sim > best_sim:
                    best_gid, best_sim = gid, sim
            if best_gid is None:
                global_id[k] = next_gid
                next_gid += 1
            else:
                global_id[k] = best_gid
        return global_id  # {(cam, local_id): global_id}
