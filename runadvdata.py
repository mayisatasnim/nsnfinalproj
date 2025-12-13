import os
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cb_dataloader import CampbellDataset
from metrics import cer, wer


datasets = [
    "GT-pairs",
    "BlurGT-Pairs",
    "iso_noise_folder",
    "PerspectiveGT-Pairs",
    "jpeg_folder"
]

model_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded on device:", device)

for data in datasets:
    print("\nMetrics for", data)

    dataset_path = os.path.join("Dataset", data)
    dataset = CampbellDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    print("Total samples:", len(dataset))

    total_cer = 0
    total_wer = 0
    count = 0

    printed_examples = 0
    max_print = 4

    for batch in loader:
        img = batch["image"]
        gt_text = batch["text"]
        sample_id = batch["id"]

        pixel_values = processor(
            images=img,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=512
            )

        pred_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        if printed_examples < max_print:
            print("\n--- Example", printed_examples + 1, "---")
            print("ID:", sample_id)
            print("GT :", gt_text)
            print("OCR:", pred_text)
            printed_examples += 1

        total_cer += cer(gt_text, pred_text)
        total_wer += wer(gt_text, pred_text)
        count += 1

    print("\n| FINAL METRICS |")
    print("Average CER:", total_cer / count)
    print("Average WER:", total_wer / count)

