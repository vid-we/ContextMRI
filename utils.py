import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import torch
import json
import lpips
from skimage.metrics import structural_similarity as compare_ssim
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def category_dict(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    category = []
    for cat in data["categories"]:
        category.append(cat["name"])

    return category

# extract the row from the dataframe that matches the filename and slice number
def extract_metadata(df, filename, slice_number):
    # Filter the dataframe for the given filename and slice number
    metadata = df[(df['filename'] == filename) & (df['slice'] == slice_number)]
    
    # If no matching records found
    if metadata.empty:
        return f"No data found for filename: {filename}, slice: {slice_number}"
    
    # Return the first matching row as a dictionary or a pandas Series
    return metadata.iloc[0]

def row_to_text_string(row, p=0.5):
    # Extract fields
    anatomy = row['anatomy']
    slice_number = row['slice']
    contrast = row['contrast']
    pathology = row['pathology']
    
    # Core string that always appears
    text = f"{anatomy}, Slice {slice_number}, {contrast}"
    
    if not pd.isna(pathology):
        # Split pathologies and count occurrences
        pathologies = pathology.split(', ')
        pathology_counts = {}
        for path in pathologies:
            pathology_counts[path] = pathology_counts.get(path, 0) + 1
        
        # Format as "N count pathology"
        numbered_pathologies = [f"{count} {path}" for path, count in pathology_counts.items()]
        text += f", Pathology: {', '.join(numbered_pathologies)}"
    
    # 50% chance to include the sequence and imaging parameters
    if random.random() < p:
        sequence = row['sequence']
        TR = row['TR']
        TE = row['TE']
        TI = row['TI']
        flip_angle = row['flip_angle']
        
        text += (f", Sequence: {sequence}, TR: {TR}, TE: {TE}, TI: {TI}, "
                 f"Flip angle: {flip_angle}")
    
    return text

# Convert CSV row into text string with appropriate format
def row_to_text_string_skm_tea(row, p=0.5):    
    # Extract fields    
    slice_number = row['slice']
    age = row['PatientAge']
    sex = row['PatientSex']
    pathology = row['pathology']

    text = f"Qdess, Knee, Slice {slice_number}, Age: {age}, Sex: {sex}"

    if not pd.isna(pathology):
        # Split pathologies and count occurrences
        pathologies = pathology.split(', ')
        pathology_counts = {}
        for path in pathologies:
            pathology_counts[path] = pathology_counts.get(path, 0) + 1
        
        # Format as "N count pathology"
        numbered_pathologies = [f"{count} {path}" for path, count in pathology_counts.items()]
        text += f", Pathology: {', '.join(numbered_pathologies)}"

    if random.random() < p:
        tr = row['RepetitionTime']
        te = row['EchoTime1']
        fa = row['FlipAngle']
        text += f", TR: {tr}, TE: {te}, Flip Angle: {fa}"    
   
    return text

def calculate_ssim(image1, image2):

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for SSIM calculation.")
    ssim, _ = compare_ssim(image1, image2, full=True, data_range=2.0)
    return ssim

def calculate_lpips(image1_np, image2_np, device):

    model = lpips.LPIPS(net='vgg').to(device)  # Use VGG backbone

    # Ensure the input is float32
    image1_np = image1_np.astype(np.float32)
    image2_np = image2_np.astype(np.float32)
    
    # Convert NumPy arrays to PyTorch tensors and add batch and channel dimensions
    image1_tensor = torch.tensor(image1_np).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 320, 320)
    image2_tensor = torch.tensor(image2_np).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 320, 320)
    
    # Normalize to [-1, 1] as required by LPIPS
    image1_tensor = (image1_tensor * 2) - 1
    image2_tensor = (image2_tensor * 2) - 1
    
    # Calculate LPIPS score
    lpips_score = model(image1_tensor, image2_tensor)
    
    return lpips_score.item()

def batch_update_json(filepath, new_data_list):

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        with open(filepath, "r") as f:
            data = json.load(f)  # Load JSON
    else:
        data = [] 

    data.extend(new_data_list)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)



# Convert user_input to text_string with appropriate format
def user_input_to_text_string(anatomy, slice_number, contrast, pathology=None, sequence=None, TR=None, TE=None, TI=None, flip_angle=None):

    text = f"{anatomy}, Slice {slice_number}, {contrast}"
    
    if pathology is not None:
        text += f", Pathology: {pathology}"
    
    if sequence is not None and TR is not None and TE is not None and TI is not None and flip_angle is not None:      
        text += (f", {sequence}, TR: {TR}, TE: {TE}, TI: {TI}, "
                 f"Flip angle: {flip_angle}")
    
    return text

def save_image(image, path):
    plt.imsave(path, image, cmap="gray")

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def count_entries_in_json(file_path):

    if not os.path.exists(file_path):
        return 0

    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Count entries based on the data structure
    if isinstance(data, dict):
        return len(data)
    elif isinstance(data, list):
        return len(data)
    else:
        raise ValueError("Unsupported JSON structure. It must be a list or a dictionary.")
        
# DW
# schema-mode counterpart to user_input_to_text_string(): same fields, typed
# dict instead of a formatted string, for SchemaConditioner (utils.py) /
# CallableConditioner (pipeline_mri.py). `pathology` here is a list[str]
# (or None), matching row_to_metadata_dict()'s "pathologies" convention -
# NOT a pre-joined string like user_input_to_text_string() takes.
def user_input_to_metadata_dict(anatomy, slice_number, contrast, pathology=None, sequence=None,
                                 TR=None, TE=None, TI=None, flip_angle=None):
    return {
        "anatomy": anatomy,
        "slice": float(slice_number),
        "contrast": contrast,
        "pathologies": list(pathology) if pathology else [],
        "sequence": sequence,
        "TR": TR, "TE": TE, "TI": TI, "flip_angle": flip_angle,
    }
    
def _pathology_counts(pathology):
    """Split a comma-separated pathology string into an ordered {name: count} dict.
    Shared by every serializer (plain / XML / schema) so that all three arms see
    exactly the same underlying information, differing only in how it is packaged.
    """
    if pd.isna(pathology):
        return {}
    pathologies = pathology.split(', ')
    counts = {}
    for path in pathologies:
        counts[path] = counts.get(path, 0) + 1
    return counts


def _xml_escape(value):
    # Minimal escaping - metadata fields are short clinical tokens, not free text,
    # but defensively escape the five XML special characters if they ever appear.
    text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&apos;"))


def row_to_text_string_xml(row, p=0.5):
    """XML-structured equivalent of row_to_text_string(). Same fields, same
    random parameter-dropout behaviour (governed by p), only the string
    serialization changes. This is the 'Arm A' (keep CLIP, reformat string)
    variant for the fastMRI knee/brain metadata.
    """
    anatomy = row['anatomy']
    slice_number = row['slice']
    contrast = row['contrast']
    pathology = row['pathology']

    parts = [
        f"<anatomy>{_xml_escape(anatomy)}</anatomy>",
        f"<slice>{_xml_escape(slice_number)}</slice>",
        f"<contrast>{_xml_escape(contrast)}</contrast>",
    ]

    counts = _pathology_counts(pathology)
    if counts:
        items = "".join(
            f'<item count="{c}">{_xml_escape(name)}</item>' for name, c in counts.items()
        )
        parts.append(f"<pathology>{items}</pathology>")

    if random.random() < p:
        parts.append(
            "<params>"
            f"<sequence>{_xml_escape(row['sequence'])}</sequence>"
            f"<TR>{_xml_escape(row['TR'])}</TR>"
            f"<TE>{_xml_escape(row['TE'])}</TE>"
            f"<TI>{_xml_escape(row['TI'])}</TI>"
            f"<flip_angle>{_xml_escape(row['flip_angle'])}</flip_angle>"
            "</params>"
        )

    return "<mri>" + "".join(parts) + "</mri>"


def row_to_text_string_skm_tea_xml(row, p=0.5):
    """XML-structured equivalent of row_to_text_string_skm_tea() ('Arm A' for
    the SKM-TEA metadata: sequence, anatomy, slice, age, sex, pathology, TR/TE/FA).
    """
    slice_number = row['slice']
    age = row['PatientAge']
    sex = row['PatientSex']
    pathology = row['pathology']

    parts = [
        "<sequence>Qdess</sequence>",
        "<anatomy>Knee</anatomy>",
        f"<slice>{_xml_escape(slice_number)}</slice>",
        f"<age>{_xml_escape(age)}</age>",
        f"<sex>{_xml_escape(sex)}</sex>",
    ]

    counts = _pathology_counts(pathology)
    if counts:
        items = "".join(
            f'<item count="{c}">{_xml_escape(name)}</item>' for name, c in counts.items()
        )
        parts.append(f"<pathology>{items}</pathology>")

    if random.random() < p:
        parts.append(
            "<params>"
            f"<TR>{_xml_escape(row['RepetitionTime'])}</TR>"
            f"<TE>{_xml_escape(row['EchoTime1'])}</TE>"
            f"<flip_angle>{_xml_escape(row['FlipAngle'])}</flip_angle>"
            "</params>"
        )

    return "<mri>" + "".join(parts) + "</mri>"


# Registry used by dataset_mri.py / train_mri.py / inference.py so the prompt
# serialization is picked with a single --prompt_format flag instead of
# scattering if/else branches across the codebase.
PROMPT_FORMATTERS = {
    "fastmri": {"plain": row_to_text_string, "xml": row_to_text_string_xml},
    "skm-tea": {"plain": row_to_text_string_skm_tea, "xml": row_to_text_string_skm_tea_xml},
}


def format_metadata(row, dataset="fastmri", prompt_format="plain", p=0.5):
    """Single entry point for turning a metadata row into a prompt string.
    dataset: 'fastmri' or 'skm-tea'. prompt_format: 'plain' or 'xml'.
    """
    return PROMPT_FORMATTERS[dataset][prompt_format](row, p=p)


# --- Raw (non-serialized) field extraction, used by the schema-aware
# conditioner (Arm B). Unlike the string formatters above, this keeps every
# field as a typed value (str for categorical, float for continuous, list[str]
# for pathology) so a learned encoder can embed each field on its own terms
# instead of re-tokenizing a formatted string. The same p-driven dropout of
# the MR parameter block is preserved so all three arms are trained under an
# identical missingness regime.
def row_to_metadata_dict(row, p=0.5):
    anatomy = row['anatomy']
    slice_number = row['slice']
    contrast = row['contrast']
    pathology = row['pathology']

    meta = {
        "anatomy": str(anatomy),
        "slice": float(slice_number),
        "contrast": str(contrast),
        "pathologies": list(_pathology_counts(pathology).keys()),  # possibly []
        "sequence": None,
        "TR": None, "TE": None, "TI": None, "flip_angle": None,
    }

    if random.random() < p:
        meta["sequence"] = str(row['sequence'])
        meta["TR"] = float(row['TR'])
        meta["TE"] = float(row['TE'])
        meta["TI"] = float(row['TI'])
        meta["flip_angle"] = float(row['flip_angle'])

    return meta


def row_to_metadata_dict_skm_tea(row, p=0.5):
    slice_number = row['slice']
    age = row['PatientAge']
    sex = row['PatientSex']
    pathology = row['pathology']

    meta = {
        "anatomy": "Knee",
        "slice": float(slice_number),
        "sequence": "Qdess",
        "age": float(age) if not pd.isna(age) else None,
        "sex": str(sex) if not pd.isna(sex) else None,
        "pathologies": list(_pathology_counts(pathology).keys()),
        "TR": None, "TE": None, "flip_angle": None,
    }

    if random.random() < p:
        meta["TR"] = float(row['RepetitionTime'])
        meta["TE"] = float(row['EchoTime1'])
        meta["flip_angle"] = float(row['FlipAngle'])

    return meta
# end