"""Load a CharCRNN checkpoint and transcribe one line image."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from arabic_ocr_bench.charctc.charset import Charset
from arabic_ocr_bench.charctc.decode import greedy_from_logits
from arabic_ocr_bench.charctc.preprocess import line_to_array, time_steps
from arabic_ocr_bench.training.checkpoints import latest_checkpoint


class CharCTCRecognizer:
    def __init__(self, ckpt_dir: str | Path, *, device: str = "cpu") -> None:
        import torch

        from arabic_ocr_bench.charctc.model import CharCRNN

        self.device = torch.device(device)
        ckpt = Path(ckpt_dir)
        if (ckpt / "model.pt").exists():
            root = ckpt
        else:
            latest = latest_checkpoint(ckpt)
            if not latest:
                raise FileNotFoundError(f"no checkpoint under {ckpt}")
            root = Path(latest)
        charset_path = root / "charset.json"
        if not charset_path.exists():
            charset_path = ckpt / "charset.json"
        self.charset = Charset.load(charset_path)
        self.model = CharCRNN(self.charset.n_classes)
        try:
            state = torch.load(
                root / "model.pt", map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(root / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.ckpt = root
        self.blank_penalty = 0.0
        for state_json in (root / "train_state.json", Path(ckpt_dir) / "train_state.json"):
            if state_json.exists():
                extra = json.loads(state_json.read_text(encoding="utf-8"))
                self.blank_penalty = float(extra.get("blank_penalty") or 0.0)
                break

    def transcribe(self, image: Image.Image) -> str:
        import torch

        arr = line_to_array(image)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        t = time_steps(arr.shape[-1])
        with torch.no_grad():
            logits = self.model(tensor)
            hyps = greedy_from_logits(
                logits, [t], self.charset, blank_penalty=self.blank_penalty
            )
        return hyps[0]
