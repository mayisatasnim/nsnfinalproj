import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cb_dataloader import CampbellDataset
from metrics import cer, wer
from PIL import Image
from tqdm import tqdm



dataset = CampbellDataset("Dataset/GT-pairs")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

print("Total samples:", len(dataset))

model_name = "microsoft/trocr-base-stage1"  # lightweight TrOCR model
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Model loaded on device:", device)


total_cer = 0
total_wer = 0
count = 0

for batch in tqdm(loader, total=len(loader), desc="Evaluating"):
    img = batch["image"]         # PIL Image
    gt_text = batch["text"]      # Ground truth
    sample_id = batch["id"]

    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values, max_length=512)
    pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    sample_cer = cer(gt_text, pred_text)
    sample_wer = wer(gt_text, pred_text)

    total_cer += sample_cer
    total_wer += sample_wer
    count += 1


print("\n| FINAL METRICS |")
print("Average CER:", total_cer / count)
print("Average WER:", total_wer / count)

