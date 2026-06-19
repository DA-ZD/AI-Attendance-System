"""
AdaFace IR50 — Direct PyTorch inference wrapper
No ONNX conversion needed.
Run from lastWork/:
    python test_adaface.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pytorch_lightning

# ── Architecture ──────────────────────────────────────────

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes*4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*4))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class IResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.prelu  = nn.PReLU(64)
        self.layer1 = self._make_layer(64,  3, stride=2)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.bn2    = nn.BatchNorm2d(2048)
        self.drop   = nn.Dropout(p=0.4)
        self.fc     = nn.Linear(32768, 512)
        self.features = nn.BatchNorm1d(512)

    def _make_layer(self, planes, n, stride=1):
        layers = [Bottleneck(self.in_planes, planes, stride)]
        self.in_planes = planes * 4
        for _ in range(1, n):
            layers.append(Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.drop(x)
        # Flatten — use adaptive pool to get 4x4
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.features(x)
        return x


class AdaFaceWrapper:
    """
    Drop-in replacement for InsightFace ArcFace.
    Returns normalized 512-dim embedding from a BGR face image (112x112).
    """
    def __init__(self, ckpt_path="adaface_ir50_ms1mv2.ckpt"):
        print(f"[AdaFace] Loading from {ckpt_path}...")
        torch.serialization.add_safe_globals([
            pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        ])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        clean = {}
        for k, v in state_dict.items():
            key = k.replace("model.", "").replace("module.", "")
            clean[key] = v

        self.model = IResNet50()
        self.model.eval()

        # Fix FC size from checkpoint
        for k, v in clean.items():
            if "fc.weight" in k:
                in_f, out_f = v.shape[1], v.shape[0]
                self.model.fc = nn.Linear(in_f, out_f)
                break

        missing, unexpected = self.model.load_state_dict(clean, strict=False)
        print(f"[AdaFace] Ready — missing={len(missing)} unexpected={len(unexpected)}")

    @torch.no_grad()
    def get_embedding(self, img_bgr_112):
        """
        Input: BGR image 112x112 (numpy uint8)
        Output: normalized 512-dim numpy array
        """
        img = cv2.cvtColor(img_bgr_112, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        emb = self.model(tensor)
        emb = emb / emb.norm(dim=1, keepdim=True)
        return emb.squeeze(0).numpy()


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    wrapper = AdaFaceWrapper("adaface_ir50_ms1mv2.ckpt")

    # Test with random image
    dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = wrapper.get_embedding(dummy)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding norm:  {np.linalg.norm(emb):.4f} (should be ~1.0)")

    # Test similarity
    emb2 = wrapper.get_embedding(dummy)
    sim = np.dot(emb, emb2)
    print(f"Same image similarity: {sim:.4f} (should be ~1.0)")

    print("\n[OK] AdaFace wrapper ready!")
    print("File saved as: adaface_wrapper.py")
