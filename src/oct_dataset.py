"""
Dataset module for Kermany OCT classification (CNV, DME, DRUSEN, NORMAL).
Loads from the folder-based Kermany dataset structure.
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data_utils import (
    get_oct_train_transforms,
    get_val_transforms,
    ImageClassificationDataset,
    HFImageClassificationDataset,
    compute_class_weights,
    get_weighted_sampler,
)


# Kermany OCT classes (alphabetical order matching folder names)
OCT_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
OCT_NUM_CLASSES = len(OCT_CLASSES)


def load_oct_dataset(data_dir):
    """
    Load Kermany OCT dataset from its folder structure.

    Expected structure (from kagglehub download of paultimothymooney/kermany2018):
        data_dir/
            OCT2017/
                train/
                    CNV/
                    DME/
                    DRUSEN/
                    NORMAL/
                test/
                    CNV/
                    DME/
                    DRUSEN/
                    NORMAL/
                val/
                    ...

    Also handles the case where train/test folders are directly in data_dir.

    Returns:
        train_df, test_df — DataFrames with columns: image_path, label, label_name
    """
    # Find the directory containing train/ with class subfolders
    # Handles varying nesting: data_dir/train/, data_dir/OCT2017/train/, etc.
    train_dir = None
    test_dir = None

    for root, dirs, files in os.walk(data_dir):
        if "train" in dirs:
            candidate = os.path.join(root, "train")
            subdirs = set(os.listdir(candidate)) if os.path.isdir(candidate) else set()
            if {"CNV", "DME", "DRUSEN", "NORMAL"}.issubset(subdirs):
                train_dir = candidate
                test_candidate = os.path.join(root, "test")
                if os.path.isdir(test_candidate):
                    test_dir = test_candidate
                break

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find train/ directory with CNV/DME/DRUSEN/NORMAL subfolders in {data_dir}. "
            "Download the dataset: kagglehub.dataset_download('paultimothymooney/kermany2018')"
        )

    def _load_split(split_dir):
        records = []
        for class_idx, class_name in enumerate(OCT_CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: class directory not found: {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    records.append({
                        "image_path": os.path.join(class_dir, fname),
                        "label": class_idx,
                        "label_name": class_name,
                    })
        return pd.DataFrame(records)

    train_df = _load_split(train_dir)
    test_df = _load_split(test_dir) if test_dir and os.path.isdir(test_dir) else pd.DataFrame()

    print(f"Loaded OCT dataset — Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"\nTrain class distribution:\n{train_df['label_name'].value_counts()}")
    if len(test_df) > 0:
        print(f"\nTest class distribution:\n{test_df['label_name'].value_counts()}")

    return train_df, test_df


def create_oct_hf_datasets(data_dir, image_size=224, val_split=0.1, seed=42):
    """
    Create train/val/test HuggingFace-compatible datasets from Kermany OCT data.

    Uses the provided train/test split. Carves a validation set from the training data
    (the original val set is too small — ~8 images per class).

    Returns:
        train_dataset, val_dataset, test_dataset, class_weights
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = load_oct_dataset(data_dir)

    # Carve validation set from training data
    train_df, val_df = train_test_split(
        train_df, test_size=val_split, stratify=train_df["label"], random_state=seed
    )

    print(f"\nFinal splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = HFImageClassificationDataset(train_df, transform=get_oct_train_transforms(image_size))
    val_dataset = HFImageClassificationDataset(val_df, transform=get_val_transforms(image_size))
    test_dataset = HFImageClassificationDataset(test_df, transform=get_val_transforms(image_size))

    train_labels = train_df["label"].values
    class_weights = compute_class_weights(train_labels, OCT_NUM_CLASSES)

    return train_dataset, val_dataset, test_dataset, class_weights


def create_oct_dataloaders(data_dir, batch_size=32, image_size=224, val_split=0.1, num_workers=4, seed=42):
    """
    Create train/val/test DataLoaders from Kermany OCT data.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = load_oct_dataset(data_dir)

    train_df, val_df = train_test_split(
        train_df, test_size=val_split, stratify=train_df["label"], random_state=seed
    )

    print(f"\nFinal splits — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = ImageClassificationDataset(train_df, transform=get_oct_train_transforms(image_size))
    val_dataset = ImageClassificationDataset(val_df, transform=get_val_transforms(image_size))
    test_dataset = ImageClassificationDataset(test_df, transform=get_val_transforms(image_size))

    train_labels = train_df["label"].values
    class_weights = compute_class_weights(train_labels, OCT_NUM_CLASSES)
    sampler = get_weighted_sampler(train_labels, OCT_NUM_CLASSES)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
