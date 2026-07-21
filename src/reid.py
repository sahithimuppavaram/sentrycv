"""
reid.py  --  cross-camera re-identification.

Turn each person's saved crops into an embedding, then match the same person
across cameras by cosine similarity -> assign one GLOBAL id. This is the piece
that mirrors Verkada's "track objects across cameras" (People Analytics).

Setup: pip install torchreid  (downloads OSNet weights on first use).
"""
import os
import glob
import numpy as np
from collections import defaultdict


class ReID:
    def __init__(self, threshold: float = 0.55):
        from torchreid.utils import FeatureExtractor
        self.extractor = FeatureExtractor(model_name="osnet_x0_25", device="cpu")
        self.threshold = threshold

    def _embed_track(self, crop_paths):
        # average the embeddings of a track's crops -> one robust vector
        feats = self.extractor(crop_paths).cpu().numpy()
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
                gallery[(cam, local_id)] = self._embed_track(paths)
        return gallery

    def assign_global_ids(self, gallery):
        """Greedy cosine matching across cameras -> global id per track."""
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
