import pickle, gzip
import pandas as pd
import numpy as np
import re

path = "data/phoenix14t.pami0.train.annotations_only.gzip"

with gzip.open(path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print("Total entries:", len(data))

# Show a few samples safely
for i in range(3):
    print(f"\nSample {i+1}:")
    print(data[i])

import pandas as pd
df = pd.DataFrame(data)
print(df.head())

# ===============================================================
# Step 3: Load all splits (Train, Dev, Test)
# ===============================================================

def load_phoenix_split(path):
    """Load one Phoenix dataset split as a pandas DataFrame."""
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {path.split('/')[-1]} with {len(df)} samples")
    return df

# Paths to all splits
train_path = "data/phoenix14t.pami0.train.annotations_only.gzip"
dev_path   = "data/phoenix14t.pami0.dev.annotations_only.gzip"
test_path  = "data/phoenix14t.pami0.test.annotations_only.gzip"

# Load all
df_train = load_phoenix_split(train_path)
df_dev   = load_phoenix_split(dev_path)
df_test  = load_phoenix_split(test_path)

# Print summary
print("\n--- Dataset Overview ---")
print("Train size:", len(df_train))
print("Dev size:", len(df_dev))
print("Test size:", len(df_test))
print("\nSample columns:", df_train.columns.tolist())

# show an example
print("\nExample from dev set:")
print(df_dev.iloc[0])


# ✅ Use the DataFrames you already created
datasets = {"train": df_train, "dev": df_dev, "test": df_test}

def clean_text(text):
    """Lowercase, remove punctuation, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", "", text)  # keep German letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_df(df):
    """Clean gloss and text columns, and add tokenized versions."""
    df = df.copy()
    df["gloss_clean"] = df["gloss"].apply(clean_text)
    df["text_clean"] = df["text"].apply(clean_text)
    df["gloss_tokens"] = df["gloss_clean"].apply(lambda x: x.split())
    df["text_tokens"] = df["text_clean"].apply(lambda x: x.split())
    return df

# Apply preprocessing
df_train_prep = preprocess_df(df_train)
df_dev_prep = preprocess_df(df_dev)
df_test_prep = preprocess_df(df_test)

# Inspect sample
print("✅ Sample preprocessed pair:\n")
print("Gloss:", df_train_prep.iloc[0]["gloss_clean"])
print("Text:", df_train_prep.iloc[0]["text_clean"])
print("\nGloss tokens:", df_train_prep.iloc[0]["gloss_tokens"])
print("Text tokens:", df_train_prep.iloc[0]["text_tokens"])

# Save to CSV (optional)
df_train_prep.to_csv("phoenix_train_clean.csv", index=False)
df_dev_prep.to_csv("phoenix_dev_clean.csv", index=False)
df_test_prep.to_csv("phoenix_test_clean.csv", index=False)

print("\n✅ Cleaned and saved all splits.")




"""
df = pd.DataFrame(data)
df.to_csv("data/annotations_train_converted.csv", index=False)
print("Saved CSV with", len(df), "rows and columns:", list(df.columns))



# Path to one sample file
feature_path = "New folder/mediapipe_features_how2sign/mediapipe_features/train/_-adcxjm1R4_0-8-rgb_front.npy"

# Load the .npy file
features = np.load(feature_path)

print("✅ Loaded successfully!")
print("Shape:", features.shape)
print("Data type:", features.dtype)
print("First 2 frames sample:")
print(features[:2])
"""