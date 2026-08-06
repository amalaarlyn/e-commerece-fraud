"""
demo.py

Runs the full Image Intelligence pipeline end-to-end:

  1. Programmatically synthesizes 4 sample test images directly in script (using PIL/OpenCV):
     - Test Image 1: Original product damage photo (Baseline claim)
     - Test Image 2: Exact duplicate copy (Resubmitted under a different customer/claim)
     - Test Image 3: Slightly edited & spliced copy (Cropped/rotated with added splice artifact)
     - Test Image 4: Unrelated distinct damage photo (Legitimate separate claim)
  2. Preprocesses images and populates the reference gallery
  3. Computes image_score + explainable risk_flags for each uploaded claim
  4. Evaluates risk flags and scores against expected fraud categories
  5. Exports sample JSON result contract handed to Member 3's Risk Fusion Engine & dashboard

This is the self-contained runnable script to showcase "Image Intelligence v1 working demo."
"""

import json
import os
import sys

# Ensure current script directory is in Python path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

from preprocessing import preprocess_image
from scoring import compute_image_score


def create_synthetic_test_images():
    """
    Synthesizes 4 sample test images programmatically using PIL/OpenCV without
    requiring any external network image downloads.
    """
    # -----------------------------------------------------------------------
    # Image 1: Base original damage photo (Legitimate claim baseline)
    # -----------------------------------------------------------------------
    img1 = Image.new("RGB", (600, 600), color=(40, 50, 70))
    draw = ImageDraw.Draw(img1)
    # Draw product frame box
    draw.rectangle([100, 100, 500, 500], outline=(180, 190, 200), width=4)
    # Draw crack damage lines
    draw.line([(150, 150), (250, 280), (280, 320), (380, 420)], fill=(220, 50, 50), width=5)
    draw.line([(250, 280), (320, 250)], fill=(220, 50, 50), width=3)
    # Add subtle random texture noise
    arr1 = np.array(img1)
    noise = np.random.randint(-10, 10, arr1.shape, dtype=np.int16)
    arr1 = np.clip(arr1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img1_final = Image.fromarray(arr1)

    # -----------------------------------------------------------------------
    # Image 2: Exact duplicate (Reused photo resubmitted under new claim)
    # -----------------------------------------------------------------------
    img2_final = img1_final.copy()

    # -----------------------------------------------------------------------
    # Image 3: Edited & spliced copy (Cropped/rotated with sharp pasted patch)
    # -----------------------------------------------------------------------
    arr3 = np.array(img1_final)
    # Paste a razor-sharp bright high-contrast patch (simulating digital tampering / splicing)
    cv2.rectangle(arr3, (300, 120), (450, 250), (255, 255, 255), -1)
    cv2.putText(arr3, "FAKE", (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    # Slight rotation to alter global alignment
    img3_pil = Image.fromarray(arr3).rotate(5, expand=False, fillcolor=(40, 50, 70))
    img3_final = img3_pil

    # -----------------------------------------------------------------------
    # Image 4: Unrelated distinct damage photo (Legitimate separate claim)
    # -----------------------------------------------------------------------
    img4 = Image.new("RGB", (600, 600), color=(80, 120, 90))
    draw4 = ImageDraw.Draw(img4)
    draw4.ellipse([150, 150, 450, 450], outline=(230, 230, 150), width=6)
    draw4.line([(200, 300), (400, 300)], fill=(255, 255, 255), width=4)
    arr4 = np.array(img4)
    noise4 = np.random.randint(-10, 10, arr4.shape, dtype=np.int16)
    arr4 = np.clip(arr4.astype(np.int16) + noise4, 0, 255).astype(np.uint8)
    img4_final = Image.fromarray(arr4)

    # -----------------------------------------------------------------------
    # Image 5: Standalone manipulated/spliced image (First-time edit, NO duplicate in gallery)
    # -----------------------------------------------------------------------
    img5 = Image.new("RGB", (600, 600), color=(140, 70, 50))
    draw5 = ImageDraw.Draw(img5)
    draw5.rectangle([120, 120, 480, 480], fill=(200, 120, 70), outline=(240, 200, 150), width=5)
    arr5 = np.array(img5)
    # Paste a razor-sharp, high-contrast spliced patch to create spatial edge disparity & ELA discrepancy
    cv2.rectangle(arr5, (200, 200), (400, 350), (255, 255, 255), -1)
    for x in range(210, 390, 15):
        cv2.line(arr5, (x, 210), (x, 340), (0, 0, 0), 2)
    cv2.putText(arr5, "TAMPERED", (215, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    img5_final = Image.fromarray(arr5)

    return {
        "img1_original": img1_final,
        "img2_duplicate": img2_final,
        "img3_edited": img3_final,
        "img4_unrelated": img4_final,
        "img5_standalone_tampered": img5_final,
    }


def run_demo():
    print("=" * 75)
    print("IMAGE INTELLIGENCE MODULE (RETURN FRAUD DETECTION) -- END TO END DEMO")
    print("=" * 75)

    # 1. Synthesize test images
    print("\n[1] Synthesizing 5 self-contained test images (PIL/OpenCV)...")
    synthetic_images = create_synthetic_test_images()

    # Preprocess all test images
    preprocessed = {
        name: preprocess_image(img) for name, img in synthetic_images.items()
    }

    # 2. Setup reference gallery (previously submitted claim images database)
    gallery = [
        {
            "img_data": preprocessed["img1_original"],
            "customer_id": "CUST_1001_LEGIT",
            "return_id": "RET_CLAIM_9001",
            "image_id": "IMG_9001_A.JPG",
        }
    ]
    print(f"[2] Reference claim gallery populated with {len(gallery)} historical claim image(s).")

    # 3. Test scenarios
    test_cases = [
        {
            "scenario": "Original Baseline Image (First upload)",
            "customer_id": "CUST_1001_LEGIT",
            "return_id": "RET_CLAIM_9001",
            "image_id": "IMG_9001_A.JPG",
            "img_data": preprocessed["img1_original"],
            "gallery_to_use": [],  # First submission, empty prior gallery
        },
        {
            "scenario": "Exact Duplicate Image (Reused across different customer/claim)",
            "customer_id": "CUST_4088_SUSPECT",
            "return_id": "RET_CLAIM_9822",
            "image_id": "IMG_9822_DUP.JPG",
            "img_data": preprocessed["img2_duplicate"],
            "gallery_to_use": gallery,
        },
        {
            "scenario": "Edited / Spliced Copy (Digital tampering & pasted patch)",
            "customer_id": "CUST_7712_SUSPECT",
            "return_id": "RET_CLAIM_9944",
            "image_id": "IMG_9944_TAMPERED.JPG",
            "img_data": preprocessed["img3_edited"],
            "gallery_to_use": gallery,
        },
        {
            "scenario": "Unrelated Distinct Image (Legitimate separate return claim)",
            "customer_id": "CUST_2055_LEGIT",
            "return_id": "RET_CLAIM_9105",
            "image_id": "IMG_9105_CLEAN.JPG",
            "img_data": preprocessed["img4_unrelated"],
            "gallery_to_use": gallery,
        },
        {
            "scenario": "Standalone Manipulated Image (First-time edit, NO duplicate in gallery)",
            "customer_id": "CUST_5501_SUSPECT",
            "return_id": "RET_CLAIM_9988",
            "image_id": "IMG_9988_STANDALONE_SPLICED.JPG",
            "img_data": preprocessed["img5_standalone_tampered"],
            "gallery_to_use": gallery,  # No visual similarity with img1_original in gallery
        },
    ]

    print("\n[3] Running Image Intelligence Pipeline on Test Scenarios:\n")

    results = []
    for idx, tc in enumerate(test_cases, start=1):
        res = compute_image_score(
            uploaded_img_data=tc["img_data"],
            candidate_gallery=tc["gallery_to_use"],
            customer_id=tc["customer_id"],
            return_id=tc["return_id"],
            image_id=tc["image_id"],
        )
        results.append((tc, res))

        print(f"--- Test Case #{idx}: {tc['scenario']} ---")
        print(f"  Customer ID : {res.customer_id}")
        print(f"  Return ID   : {res.return_id}")
        print(f"  Image ID    : {res.image_id}")
        print(f"  Image Score : {res.image_score:.4f}  (Range 0.0 - 1.0, Higher = Suspicious)")
        print(f"  Risk Flags ({len(res.risk_flags)}):")
        if not res.risk_flags:
            print("      - [CLEAN] No fraud flags triggered.")
        else:
            for flag in res.risk_flags:
                print(f"      - [{flag.code}] {flag.message} (severity: {flag.severity})")
        print()

    # 4. Export sample JSON contract for Member 3's Risk Fusion Engine & frontend
    top_flagged_result = max(results, key=lambda x: x[1].image_score)[1]
    output_dir = os.path.dirname(os.path.abspath(__file__))
    sample_json_path = os.path.join(output_dir, "sample_image_output.json")

    with open(sample_json_path, "w") as f:
        json.dump(top_flagged_result.to_dict(), f, indent=2)

    print(f"[4] Exported JSON contract for highest-risk image to:\n    {sample_json_path}")

    print("\nSample JSON Contract Payload (Top-level key 'customer_id' matches GraphIntelligenceResult):")
    print("-" * 75)
    print(json.dumps(top_flagged_result.to_dict(), indent=2))
    print("-" * 75)

    print("\n" + "=" * 75)
    print("DEMO COMPLETE -- IMAGE INTELLIGENCE MODULE READY")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
