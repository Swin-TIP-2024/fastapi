import os
from xml.etree import ElementTree as ET

from .features import FULL_FEATURE_LIST


# Extract Permissions from AndroidManifest.xml
def extract_permissions(manifest_path):
    tree = ET.parse(manifest_path)
    root = tree.getroot()
    permissions = []
    for perm in root.findall(".//uses-permission"):
        permissions.append(
            perm.attrib["{http://schemas.android.com/apk/res/android}name"]
        )
    return permissions


# Search for method invocations in smali files
def search_smali_files(smali_dir, features):
    matched_features = []
    for root, dirs, files in os.walk(smali_dir):
        for file in files:
            if file.endswith(".smali"):
                with open(os.path.join(root, file), "r") as f:
                    content = f.read()
                    for feature in features:
                        if feature in content:
                            matched_features.append(feature)
    return matched_features


# Main function to run feature extraction
def extract_features(apk_decompiled_dir):
    # 1. Extract permissions from AndroidManifest.xml
    manifest_path = os.path.join(apk_decompiled_dir, "AndroidManifest.xml")
    permissions = extract_permissions(manifest_path)

    # 2. Search smali files for method invocations and API calls
    smali_dir = os.path.join(apk_decompiled_dir, "smali")
    matched_features = search_smali_files(smali_dir, FULL_FEATURE_LIST)

    # Combine and return all extracted features
    return permissions + matched_features


def extract_binary_features():
    # Get extracted features
    apk_decompiled_dir = "./output"  # Path to your decompiled APK directory
    extracted_features = extract_features(apk_decompiled_dir)

    # Create a feature vector (1 if feature is present, 0 if absent)
    binary_feature_vector = [
        1 if feature in extracted_features else 0 for feature in FULL_FEATURE_LIST
    ]

    return binary_feature_vector
