# pgd_attack_simple.py - Works with your existing cb_dataloader
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
from paddleocr import PaddleOCR

from cb_dataloader import CampbellDataset

# Initialize OCR
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
print("✅ OCR model loaded!")

def ocr_predict_pil(image_pil):
    """Run OCR on PIL Image"""
    # Convert PIL to numpy
    img_np = np.array(image_pil)
    
    # Convert RGB to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Run OCR
    result = ocr.ocr(img_bgr, cls=False)
    
    if result and result[0]:
        texts = [line[1][0] for line in result[0]]
        return ' '.join(texts)
    return ""

def text_similarity(text1, text2):
    """Simple text similarity"""
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if not text1 or not text2:
        return 0.0
    
    # Simple character overlap
    set1 = set(text1)
    set2 = set(text2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def pgd_attack_image(image_pil, target_text, epsilon=0.1, steps=10):
    """
    Simple PGD attack on PIL Image
    image_pil: PIL Image
    target_text: ground truth text
    epsilon: max perturbation (0-255)
    steps: number of attack iterations
    """
    # Convert PIL to numpy
    img_np = np.array(image_pil).astype(np.float32)
    orig_np = img_np.copy()
    
    # Normalize to [0, 1] for attack
    img_norm = img_np / 255.0
    orig_norm = orig_np / 255.0
    
    alpha = epsilon / 255.0 / steps  # Step size in normalized space
    
    # Store best adversarial example
    best_img = img_norm.copy()
    best_similarity = 1.0
    
    print(f"  Running {steps} PGD steps...")
    
    for step in range(steps):
        # Convert to PIL for OCR
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        
        # Get prediction
        pred_text = ocr_predict_pil(img_pil)
        similarity = text_similarity(pred_text, target_text)
        
        # Simple gradient approximation: random perturbation
        # In real PGD, you'd compute gradient through OCR model
        # This is a simplified version
        grad = np.random.randn(*img_norm.shape) * 0.01
        
        # Attack: move away from correct prediction
        img_norm = img_norm + alpha * np.sign(grad)
        
        # Project back to epsilon ball
        delta = img_norm - orig_norm
        delta = np.clip(delta, -epsilon/255.0, epsilon/255.0)
        img_norm = orig_norm + delta
        
        # Clip to valid range
        img_norm = np.clip(img_norm, 0, 1)
        
        # Track best
        if similarity < best_similarity:
            best_similarity = similarity
            best_img = img_norm.copy()
        
        if (step + 1) % 3 == 0:
            print(f"    Step {step+1}/{steps}: similarity = {similarity:.3f}")
    
    # Convert back to PIL
    best_uint8 = (best_img * 255).astype(np.uint8)
    return Image.fromarray(best_uint8)

def run_pgd_attack(num_images=5, epsilons=[10, 30, 50]):
    """
    Run PGD attack on dataset
    epsilons: list of epsilon values (0-255 pixel values)
    """
    # Load dataset using your existing code
    ds = CampbellDataset("Dataset/GT-pairs")
    
    print(f"Total dataset size: {len(ds)}")
    print(f"Testing on first {num_images} images")
    print(f"Epsilons to test: {epsilons}")
    
    # Create output directory
    output_dir = "Dataset/PGD-Attacks"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for eps in epsilons:
        print(f"\n{'='*60}")
        print(f"Testing with epsilon = {eps} (pixel value)")
        print('='*60)
        
        correct_before = 0
        correct_after = 0
        total_tested = 0
        
        for i in range(min(num_images, len(ds))):
            sample = ds[i]
            img = sample["image"]
            target_text = sample["text"]
            img_id = sample["id"]
            
            # Get original prediction
            orig_pred = ocr_predict_pil(img)
            orig_sim = text_similarity(orig_pred, target_text)
            
            # Consider correct if similarity > 0.5
            is_correct = orig_sim > 0.5
            
            if is_correct:
                correct_before += 1
                total_tested += 1
                
                print(f"\nImage {img_id}:")
                print(f"  Target text: '{target_text}'")
                print(f"  Original OCR: '{orig_pred}' (similarity: {orig_sim:.3f})")
                
                # Run PGD attack
                adv_img = pgd_attack_image(
                    image_pil=img,
                    target_text=target_text,
                    epsilon=eps,
                    steps=10
                )
                
                # Get adversarial prediction
                adv_pred = ocr_predict_pil(adv_img)
                adv_sim = text_similarity(adv_pred, target_text)
                
                print(f"  Adversarial OCR: '{adv_pred}' (similarity: {adv_sim:.3f})")
                
                # Check if attack succeeded
                if adv_sim > 0.5:
                    correct_after += 1
                    print(f"  Result: Attack FAILED")
                else:
                    print(f"  Result: Attack SUCCESSFUL!")
                
                # Save results
                save_path = os.path.join(output_dir, f"{img_id}_eps{eps}.png")
                adv_img.save(save_path)
                
                # Save ground truth
                gt_path = os.path.join(output_dir, f"{img_id}_eps{eps}.gt.txt")
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write(target_text)
                
                print(f"  Saved to: {save_path}")
        
        # Calculate statistics
        if total_tested > 0:
            attack_success = 100 * (1 - correct_after/total_tested)
            
            print(f"\n📊 Results for ε={eps}:")
            print(f"  Tested: {total_tested} images")
            print(f"  Attack success rate: {attack_success:.1f}%")
            
            results.append({
                'epsilon': eps,
                'attack_success': attack_success,
                'tested_images': total_tested
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    print(f"{'Epsilon':<10} {'Attack Success':<15} {'Images Tested':<15}")
    print("-" * 40)
    for res in results:
        print(f"{res['epsilon']:<10} {res['attack_success']:<14.1f}% {res['tested_images']:<14}")
    
    # Save summary to file
    summary_path = os.path.join(output_dir, "attack_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PGD Attack Results\n")
        f.write("="*40 + "\n")
        for res in results:
            f.write(f"Epsilon: {res['epsilon']} | Success: {res['attack_success']:.1f}% | Tested: {res['tested_images']}\n")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"📄 Summary saved to: {summary_path}")

def main():
    # Configuration
    num_images_to_test = 5
    epsilons = [10, 25, 50, 75, 100]  # Pixel values (0-255)
    
    print("PGD Adversarial Attack on OCR")
    print("="*60)
    print(f"Dataset: Dataset/GT-pairs")
    print(f"OCR Engine: PaddleOCR")
    print(f"Number of images: {num_images_to_test}")
    print(f"Epsilon values: {epsilons}")
    
    run_pgd_attack(
        num_images=num_images_to_test,
        epsilons=epsilons
    )

if __name__ == "__main__":
    main()