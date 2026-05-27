
import os
import pandas as pd
import numpy as np
import random
import torch
from utils import row_to_text_string, row_to_text_string_skm_tea

class SKMDataset:
    def __init__(self, metadata_file, train=True):
        self.metadata = pd.read_csv(metadata_file)
        self.train = train

        valid_rows = []
        for idx, row in self.metadata.iterrows():
            anatomy = row['anatomy']
            filename = row['filename']
            echo_num = row['echo']
            slice_number = row['slice']
            if self.train:
                file_path = os.path.join(f"../skm-tea/processed_data/train/slice", filename, f"echo{echo_num}_{slice_number:03d}.npy")
            else:
                file_path = os.path.join(f"../skm-tea/processed_data/val/slice", filename, f"echo{echo_num}_{slice_number:03d}.npy")

            if os.path.exists(file_path):
                valid_rows.append(row)

        self.metadata = pd.DataFrame(valid_rows).reset_index(drop=True)

    def get_item_from_index(self, idx):

        row = self.metadata.iloc[idx]
        anatomy = row['anatomy']
        filename = row['filename']
        echo_num = row['echo']
        slice_number = row['slice']
        pathology = row['pathology']

        if self.train:
            text = row_to_text_string_skm_tea(row)
        else:
            text = row_to_text_string_skm_tea(row, p=1.0)

        if self.train:
            file_path = os.path.join(f"../skm-tea/processed_data/train/slice", filename, f"echo{echo_num}_{slice_number:03d}.npy")
        else:
            file_path_img = os.path.join(f"../skm-tea/processed_data/val/slice", filename, f"echo{echo_num}_{slice_number:03d}.npy")
            file_path_mps = os.path.join(f"../skm-tea/processed_data/val/mps", filename, f"{slice_number:03d}.npy")

        if self.train:
            if os.path.exists(file_path):
                image_data = np.load(file_path)
                real = image_data.real.astype(np.float32)
                imag = image_data.imag.astype(np.float32)    
                real = torch.from_numpy(real).float().unsqueeze(0)
                imag = torch.from_numpy(imag).float().unsqueeze(0)
                img = torch.cat([real, imag], dim=0)
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        else:
            if os.path.exists(file_path_img) and os.path.exists(file_path_mps):
                image_data = np.load(file_path_img)
                mps_data = np.load(file_path_mps)
                image_data = torch.tensor(image_data).unsqueeze(0)
                mps_data = torch.tensor(mps_data)
            else:
                raise FileNotFoundError(f"File not found: {file_path_img}")
        
            if pd.isna(pathology):
                pathology = "null"

        if self.train:
            return img, text
        else:
            return image_data, mps_data, text, filename, slice_number, pathology, anatomy

    def __getitem__(self, idx):

        if self.train:
            img, text = self.get_item_from_index(idx)
        else:
            img, mps, text, filename, slice_number, pathology, anatomy = self.get_item_from_index(idx)
            
        if self.train:
            if img.shape != (2, 512, 512):
                raise ValueError(f"Not supported Image size: {img.shape}")
        else:
            if img.shape != (1, 512, 512):
                raise ValueError(f"Not supported Image size: {img.shape}")

        if self.train:
            # dropout condition for unconditional generation
            p = random.random()
            if p < 0.1:
                text = ""
                
            return {"image": img, "prompt": text}
        else:
            return {"image": img, "mps": mps, "prompt": text, "filename": filename, "slice_number": slice_number, "pathology": pathology, "anatomy": anatomy}

    def __len__(self):
        return len(self.metadata)

class MRIDataset:
    def __init__(self, metadata_file_knee, metadata_file_brain, train=True):
        self.metadata_knee = pd.read_csv(metadata_file_knee)
        # #DW, 17.5.2026 Brain data optional - check for None
        if metadata_file_brain is not None:
            self.metadata_brain = pd.read_csv(metadata_file_brain)
        else:
            self.metadata_brain = pd.DataFrame()
        #self.metadata_brain = pd.read_csv(metadata_file_brain)
        self.train = train

        self.metadata_knee_prev = self.metadata_knee.copy()

        valid_rows_knee = []
        for idx, row in self.metadata_knee.iterrows():
            anatomy = row['anatomy']
            filename = row['filename']
            slice_number = row['slice']
            if self.train:
                #file_path = os.path.join(f"../fastmri/{anatomy}_mvue_320_train", "slice", filename, f"{slice_number:03d}.npy")
                file_path = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
            else:
                #file_path = os.path.join(f"../fastmri/{anatomy}_mvue_320_val", "slice", filename, f"{slice_number:03d}.npy")
                file_path = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
                #print(file_path)
                
            if os.path.exists(file_path):
                valid_rows_knee.append(row)

        valid_rows_brain = []
        for idx, row in self.metadata_brain.iterrows():
            anatomy = row['anatomy']
            filename = row['filename']
            slice_number = row['slice']
            if self.train:
                #file_path = os.path.join(f"../fastmri/{anatomy}_mvue_320_train", "slice", filename, f"{slice_number:03d}.npy")
                file_path = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
            else:
                #file_path = os.path.join(f"../fastmri/{anatomy}_mvue_320_val", "slice", filename, f"{slice_number:03d}.npy")
                file_path = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
                
            if os.path.exists(file_path):
                valid_rows_brain.append(row)

        self.metadata_knee = pd.DataFrame(valid_rows_knee).reset_index(drop=True)
        self.metadata_brain = pd.DataFrame(valid_rows_brain).reset_index(drop=True)
        
        # Combine two metadata into one
        self.metadata = pd.concat([self.metadata_knee, self.metadata_brain], ignore_index=True)

    def get_item_from_index(self, idx):

        row = self.metadata.iloc[idx]
        anatomy = row['anatomy']
        filename = row['filename']
        slice_number = row['slice']
        pathology = row['pathology']

        if self.train:
            text = row_to_text_string(row)
        else:
            text = row_to_text_string(row, p=1.0)
            
        # Change the path in your custom MRI data repository
        if self.train:
            #DW, 17.5.2026: file_path = os.path.join(f"../fastmri/{anatomy}_mvue_320_train", "slice", filename, f"{slice_number:03d}.npy")
            file_path = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
            
        else:
            #file_path_img = os.path.join(f"../fastmri/{anatomy}_mvue_320_val", "slice", filename, f"{slice_number:03d}.npy")
            #file_path_mps = os.path.join(f"../fastmri/{anatomy}_mvue_320_val", "mps", filename, f"{slice_number:03d}.npy")
            file_path_img = os.path.join(f"./assets/fastmri/{anatomy}", filename, "slice", f"{slice_number:03d}.npy")
            file_path_mps = os.path.join(f"./assets/fastmri/{anatomy}", filename, "mps", f"{slice_number:03d}.npy")
            print(file_path_img)
            print(file_path_mps)
     
        # The file path should be contained numpy complex64 values
        if self.train:
            if os.path.exists(file_path):
                image_data = np.load(file_path)
                real = image_data.real.astype(np.float32)
                imag = image_data.imag.astype(np.float32)    
                real = torch.from_numpy(real).float().unsqueeze(0)
                imag = torch.from_numpy(imag).float().unsqueeze(0)
                img = torch.cat([real, imag], dim=0)
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
        
        else:
            if os.path.exists(file_path_img) and os.path.exists(file_path_mps):
                image_data = np.load(file_path_img)
                mps_data = np.load(file_path_mps)
                image_data = torch.tensor(image_data).unsqueeze(0)
                mps_data = torch.tensor(mps_data)
                
        if pd.isna(pathology):
            pathology = "null"
            
        if self.train:
            return img, text
        else:
            return image_data, mps_data, text, filename, slice_number, pathology, anatomy

    def __getitem__(self, idx):

        if self.train:
            img, text = self.get_item_from_index(idx)
        else:
            img, mps, text, filename, slice_number, pathology, anatomy = self.get_item_from_index(idx)

        if self.train:
            if img.shape != (2, 320, 320):
                raise ValueError(f"Not supported Image size: {img.shape}")
        else:
            if img.shape != (1, 320, 320):
                raise ValueError(f"Not supported Image size: {img.shape}")

        if self.train:
            p = random.random()
            if p < 0.1:
                text = ""
            return {"image": img, "prompt": text}
        
        else:
            return {"image": img, "mps": mps, "prompt": text, "filename": filename, "slice_number": slice_number, "pathology": pathology, "anatomy": anatomy}

    def __len__(self):
        return len(self.metadata)