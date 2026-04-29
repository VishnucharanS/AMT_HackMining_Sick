import cv2
import numpy as np

# =========================
# CONFIG (set after testing)
# =========================
BASELINE = {
    "blur": 120.0,
    "contrast": 60.0,
    "missing_depth": 0.05,
    "depth_std": 20.0
}

# weights for final score
WEIGHTS = {
    "blur": 0.3,
    "contrast": 0.2,
    "missing": 0.3,
    "depth_std": 0.2
}


# =========================
# FEATURE FUNCTIONS
# =========================

def compute_blur(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_contrast(gray):
    return np.std(gray)


def compute_missing_depth(depth):
    return np.sum(depth == 0) / depth.size


def compute_depth_std(depth):
    # ignore zeros (invalid depth)
    valid = depth[depth > 0]
    if len(valid) == 0:
        return 0
    return np.std(valid)

# =========================
# NORMALIZATION
# =========================

def normalize_features(features):
    norm = {}

    norm["blur"] = features["blur"] / BASELINE["blur"]
    norm["contrast"] = features["contrast"] / BASELINE["contrast"]

    # inverse because higher missing = worse
    norm["missing"] = 1 - (features["missing_depth"] / BASELINE["missing_depth"])

    # inverse because higher std = worse
    norm["depth_std"] = 1 - (features["depth_std"] / BASELINE["depth_std"])

    return norm


# =========================
# SCORING
# =========================

def compute_score(norm):
    score = (
        WEIGHTS["blur"] * norm["blur"] +
        WEIGHTS["contrast"] * norm["contrast"] +
        WEIGHTS["missing"] * norm["missing"] +
        WEIGHTS["depth_std"] * norm["depth_std"]
    )

    return max(0, min(score * 100, 100))  # clamp 0–100


def classify(score):
    if score > 75:
        return "NORMAL"
    elif score > 40:
        return "REDUCED PERFORMANCE"
    else:
        return "CRITICAL"

# =========================
# MAIN PIPELINE
# =========================

def process(rgb_path, depth_path):
    # load images
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if rgb is None or depth is None:
        print("Error loading images")
        return

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # --- extract features ---
    features = {
        "blur": compute_blur(gray),
        "contrast": compute_contrast(gray),
        "missing_depth": compute_missing_depth(depth),
        "depth_std": compute_depth_std(depth)
    }

    print("\nRaw Features:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")

    # --- normalize ---
    norm = normalize_features(features)

    print("\nNormalized Features:")
    for k, v in norm.items():
        print(f"{k}: {v:.4f}")

    # --- score ---
    score = compute_score(norm)
    state = classify(score)

    print("\nFinal Result:")
    print(f"Score: {score:.2f}")
    print(f"State: {state}")

    # --- display ---
    cv2.putText(rgb, f"Score: {score:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(rgb, state, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if state == "NORMAL" else (0, 0, 255), 2)

    cv2.imshow("RGB", rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    process("rgb.png", "depth.png")