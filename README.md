README

Project Title: Safety and Accuracy Analysis of TrOCR between English-only and English-Greek Historical Data

Project description: Accurate OCR is crucial for preserving historical documents, yet its performance remains limited by document degradation, outdated typography, and multilingual content. In this work, we evaluate TrOCR, a Transformer-based OCR model, on historical data from the GT4HistComment dataset. We examine model accuracy, robustness, and safety under both clean and adversarial conditions that reflect realistic archival noise, including Gaussian blur, ISO noise, perspective distortion, JPEG compression, and synthetic shadow overlays. Our primary perturbation methods for safety and robustness analysis are the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). We compare performance on English-only inputs with bilingual English–Greek data from the Campbell subset. Using Character Error Rate (CER) and Word Error Rate (WER) as evaluation metrics, we find that TrOCR achieves near-perfect accuracy on clean English data and degrades only modestly under adversarial perturbations, with PGD producing disproportionately larger increases in both CER and WER. These results highlight both the strengths and the vulnerabilities of Transformer-based OCR models in historical and multilingual settings.

How to pull to run: 

Contributors: Mayisa Tasnim, Samuel Mbuh, Luis Arciniega, and Bilal Tariq

The Datasets folder contains folders of image-text pairs. Each are labeled with the corresponding perturbation. The clean dataset is GT-pairs.
The metric_plots folder contains graphs comparing averaged character error count (CER) and word error count (WER) for TrOCR on clean and naturally perturbed images as well as adversarial attacks.
The trocr_vit_attention_results contains 5 example saliency maps.

Scripts: 

runadvdata.py 
This scripts performs OCR using the TrOCR model on the clean dataset on the clean and naturally perturbed datasets and obtains averaged character error count (CER) and word error count (WER) for each. 
Output: output-file-4455022.out


runattack.py
This script performs FGSM and PGD attacks on clean and naturally perturbed datasets and obtains averaged CER and WER for each. 
Output: output-file-4455079.out

TrOcrBase.py
This script performs OCR using the TrOCR model on the clean GT-Pairs dataset from the Campbell data. 

Cd_dataloader.py
This script imports and stores the Campbell image data ( text images alongside ground truth labels ) in the current folder directory in a folder named Datasets. Here, we are able to choose whether to look through only English data or the full dataset with the onlyenglish variable. 

Metrics.py
Contains two methods, cer and wer, which respectively return the character and word error rates.    

Image Perturbation Scripts Overview
This directory contains several scripts that apply different adversarial-style image corruptions to PNG files. Each script follows a consistent structure: a core transformation method (e.g., add_*) and a modify function that processes a specified range of Campbell dataset images, saving both the outputs and their corresponding text labels for comparison.

ISO_noise.py
This script provides the add_iso_noise method, which takes a PNG image, a noise intensity (float), and a color noise factor (float). The noise intensity controls the level of grain added to the image, while the color factor determines how much the noise varies across channels. The modify function applies ISO-style noise to selected Campbell dataset images and stores the results in a dedicated folder.

blur_adversarial.py
Contains the add_blur method, which applies a blur of adjustable severity to obscure edges and fine details. The function takes the PNG image as its parameter. The Gaussian blur radius is selected randomly between 0 and 3. The modify function applies the blur to the selected images and saves the corrupted outputs along with their labels.

perspective_adversarial.py
Includes the add_perspective_shift method, which introduces adversarial perspective distortion based on a distortion angle selected randomly between -4 to 4. This transformation warps the image as if viewed from an altered angle. The modify function applies the distortion to a range of images, storing the results and labels.

shadow_adversarial.py
Provides the add_shadow_adversarial method, which overlays synthetic shadows at random positions on the image to obscure image features. The modify function applies shadow-based corruption and saves the outputs and labels.

to_jpeg.py
Offers the convert_to_jpeg method, which re-encodes a PNG image as JPG at a given quality level (0–100). Lower-quality values introduce stronger compression artifacts, such as blocking and color noise. The modify function converts the selected images to JPEG format and stores the compressed images along with their labels.
 
 

