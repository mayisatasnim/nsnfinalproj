import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from cb_dataloader import CampbellDataset
from metrics import cer, wer
from PIL import Image


datasets = [
    "GT-pairs",
    "BlurGT-Pairs",
    "iso_noise_folder",
    "PerspectiveGT-Pairs",
    "jpeg_folder",
    "ShadowedGT-Pairs"
]

model_name = "microsoft/trocr-base-stage1"

epsilon = 0.01
pgd_step_size = 0.003
pgd_iters = 10

#only for first 5 pics
NUM_VIS = 5

output_root = "trocr_vit_attention_results_multilingual"

device = "cuda" if torch.cuda.is_available() else "cpu"

#model
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

#for loss
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

if hasattr(model.config, "loss_type"):
    model.config.loss_type = "ForCausalLMLoss"


if hasattr(model, "set_attn_implementation"):
    model.set_attn_implementation("eager")
elif hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "eager"

model.config.output_attentions = True

model.to(device)
model.eval()

print("Model loaded on device:", device)
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))


@torch.no_grad()
def decode(pixel_values):
    ids = model.generate(pixel_values, max_length=512)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def fgsm(pixel_values, gt_text):
    x = pixel_values.clone().detach().requires_grad_(True)
    labels = processor.tokenizer(gt_text, return_tensors="pt").input_ids.to(device)

    model.zero_grad(set_to_none=True)
    loss = model(pixel_values=x, labels=labels).loss
    loss.backward()

    adv = torch.clamp(x + epsilon * x.grad.sign(), 0, 1).detach()
    return adv

def pgd(pixel_values, gt_text):
    x0 = pixel_values.clone().detach()
    adv = x0.clone()

    labels = processor.tokenizer(gt_text, return_tensors="pt").input_ids.to(device)

    for _ in range(pgd_iters):
        adv.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        loss = model(pixel_values=adv, labels=labels).loss
        loss.backward()

        adv = adv + pgd_step_size * adv.grad.sign()
        adv = torch.max(torch.min(adv, x0 + epsilon), x0 - epsilon)
        adv = torch.clamp(adv, 0, 1).detach()

    return adv

def saliency(pixel_values):
    
    # Returns (S,S) numpy in [0,1], typically 14x14 for ViT-base.
    
    with torch.no_grad():
        out = model.encoder(pixel_values, output_attentions=True, return_dict=True)
        attns = out.attentions
        if attns is None:
            raise RuntimeError(
                "error w/ attention \n"
            )

        A = torch.stack([a.mean(dim=1) for a in attns], dim=0)[:, 0]

        T = A.size(-1)
        I = torch.eye(T, device=A.device).unsqueeze(0)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)

        rollout = A[0]
        for i in range(1, A.size(0)):
            rollout = A[i] @ rollout

        cls_to_patches = rollout[0, 1:]  # drop CLS self
        n = cls_to_patches.numel()
        s = int(n ** 0.5)

        sal = cls_to_patches.reshape(s, s)
        sal = sal / (sal.max() + 1e-12)
        return sal.cpu().numpy()

def overlay_saliency(original_pil, saliency_patch):
    img = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    sal_resized = cv2.resize(saliency_patch, (w, h), interpolation=cv2.INTER_LINEAR)
    sal_uint8 = (sal_resized * 255).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)

    return cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)

def save_model_input_image(pixel_values, path):
    t = pixel_values.detach().cpu().squeeze(0)  # (3,224,224)
    t = (t * 255).clamp(0, 255).byte()
    arr = t.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


os.makedirs(output_root, exist_ok=True)

for data in datasets:
    print("\nDataset:", data)

    dataset_path = os.path.join("Dataset", data)
    dataset = CampbellDataset(dataset_path)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    out_dir = os.path.join(output_root, data)
    os.makedirs(out_dir, exist_ok=True)

    cer_clean = wer_clean = 0.0
    cer_fgsm = wer_fgsm = 0.0
    cer_pgd = wer_pgd = 0.0
    count = 0

    for i, batch in enumerate(tqdm(loader, desc=data, total=len(dataset), ncols=90)):
        img = batch["image"]      # PIL original (line image)
        gt_text = batch["text"]
        sample_id = batch["id"]

        # preprocess 
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)

        # clean
        pred_clean = decode(pixel_values)
        cer_clean += cer(gt_text, pred_clean)
        wer_clean += wer(gt_text, pred_clean)

        # attacks
        pv_fgsm = fgsm(pixel_values, gt_text)
        pv_pgd = pgd(pixel_values, gt_text)

        pred_fgsm = decode(pv_fgsm)
        pred_pgd = decode(pv_pgd)

        cer_fgsm += cer(gt_text, pred_fgsm)
        wer_fgsm += wer(gt_text, pred_fgsm)

        cer_pgd += cer(gt_text, pred_pgd)
        wer_pgd += wer(gt_text, pred_pgd)

        count += 1

        #saliency maps
        if i < NUM_VIS:
            sal_clean = saliency(pixel_values)
            sal_fgsm = saliency(pv_fgsm)
            sal_pgd = saliency(pv_pgd)

            cv2.imwrite(os.path.join(out_dir, f"{sample_id}_attn_clean.png"),
                        overlay_saliency(img, sal_clean))
            cv2.imwrite(os.path.join(out_dir, f"{sample_id}_attn_fgsm.png"),
                        overlay_saliency(img, sal_fgsm))
            cv2.imwrite(os.path.join(out_dir, f"{sample_id}_attn_pgd.png"),
                        overlay_saliency(img, sal_pgd))

            #what the model actually sees (224x224)
            save_model_input_image(pixel_values, os.path.join(out_dir, f"{sample_id}_modelinput_clean.png"))
            save_model_input_image(pv_fgsm, os.path.join(out_dir, f"{sample_id}_modelinput_fgsm.png"))
            save_model_input_image(pv_pgd, os.path.join(out_dir, f"{sample_id}_modelinput_pgd.png"))

    print(f"Samples: {count}")
    print("Average CER (clean):", cer_clean / count)
    print("Average WER (clean):", wer_clean / count)
    print("Average CER (FGSM): ", cer_fgsm / count)
    print("Average WER (FGSM): ", wer_fgsm / count)
    print("Average CER (PGD):  ", cer_pgd / count)
    print("Average WER (PGD):  ", wer_pgd / count)

print("\nDone. Results saved to:", output_root)

