import os
import re
from torch.utils.data import Dataset
from PIL import Image


ENGLISH_REGEX = re.compile(r"^[A-Za-z0-9\s.,;:'\"!?()\-\–—]+$")

def is_english_only(text):
    text = text.strip()
    return bool(ENGLISH_REGEX.match(text)) and len(text) > 0


class CampbellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for fname in sorted(os.listdir(self.root_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            sample_id = os.path.splitext(fname)[0]
            txt_path = os.path.join(self.root_dir, sample_id + ".gt.txt")

            if not os.path.exists(txt_path):
                continue
            # Read GT once
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Always enforce English-only
            if not is_english_only(text):
                continue

            self.samples.append(sample_id)

        print(f"[{os.path.basename(root_dir)}] English-only samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # Locate image
        for ext in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(self.root_dir, sample_id + ext)
            if os.path.exists(img_path):
                break

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        txt_path = os.path.join(self.root_dir, sample_id + ".gt.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "id": sample_id,
            "image": image,
            "text": text
        }

             
