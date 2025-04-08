import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
import math

class NoiseData(Dataset):
    def __init__(self, dir='../data', filename='ddata_final_fft_0318.xlsx', use_type=None, transform=None, use_bowl=True):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        self.use_bowl = use_bowl
        try:
            if 'train' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_train_0318.xlsx'), sheet_name=None)
            elif 'test' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_test_0318.xlsx'), sheet_name=None)
            else:
                all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        self.sheet_indices = []
        
        for sheet_idx, (sheet_name, df) in enumerate(all_sheets.items()):
            self.sheet_names.append(sheet_name)
            df['sheet_idx'] = sheet_idx
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            # print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx], self.dataFrame[keys[5]][idx], self.dataFrame[keys[6]][idx], self.dataFrame[keys[7]][idx]]
        output = [self.dataFrame[keys[8]][idx]]
        sheet_idx = self.dataFrame['sheet_idx'][idx]
    
        if self.use_bowl:
            bowl = torch.tensor([self.dataFrame[keys[4]][idx]], dtype=torch.long)
        else:
            bowl = torch.tensor([0], dtype=torch.long)

        if self.transform is not None:
            input = self.transform(input)

        input = torch.tensor(input, dtype=torch.float32).clone().detach()
        output = torch.tensor(output, dtype=torch.float32).clone().detach()

        if self.use_type:
            type_ = torch.LongTensor([self.dataFrame[keys[0]][idx]])
            return input, output, type_, bowl, sheet_idx
        else:
            return input, output, bowl, sheet_idx
    
class NoiseDataFiltered(Dataset):
    def __init__(self):
        pass
    
class NoiseDataBin(Dataset):
    def __init__(self, dir='../data', filename='data_final.xlsx', use_type = None, transform = None, num_bins=51):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        self.num_bins = num_bins
        try:
            all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        for sheet_name, df in all_sheets.items():
            self.sheet_names.append(sheet_name)
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx]]
        output = [self.dataFrame[keys[len(keys)-1]][idx]]
        if self.transform is not None:
            input = self.transform(input)
        # Bin values
        bins = np.array(range(20, 70, 1))   # num_bins = len(bins) + 1  51
        bin_output = torch.LongTensor(np.digitize(output, bins))

        bins1 = np.array(range(20, 70, 3)) # num_bins = len(bins) + 1  18
        bin_output0 = torch.LongTensor(np.digitize(output, bins1))

        bins1 = np.array(range(20, 70, 11)) # num_bins = len(bins) + 1  6
        bin_output1 = torch.LongTensor(np.digitize(output, bins1))

        bins1 = np.array(range(20, 70, 24)) # num_bins = len(bins) + 1  4
        bin_output2 = torch.LongTensor(np.digitize(output, bins1))

        input = torch.tensor(input).to(torch.float32)
        output = torch.tensor(output).to(torch.float32)
        if self.use_type:
            type_ = torch.LongTensor(np.array(range(self.dataFrame[keys[0]][idx]*self.num_bins, (self.dataFrame[keys[0]][idx]+1)*self.num_bins)))
            return input, output, bin_output, bin_output0, bin_output1, bin_output2, type_
        else:
            return input, output, bin_output, bin_output0, bin_output1, bin_output2

class NoiseDataFFT(Dataset):
    def __init__(self, dir='../data', filename='data_final_fft_0318.xlsx', use_type = None, transform = None, debug = None, use_bowl=True, fft_out=26):
        self.dir = dir
        self.filename = filename
        self.use_type = use_type
        self.transform = transform
        self.debug = debug
        self.use_bowl = use_bowl
        self.fft_out = fft_out
        try:
            if 'train' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_train_0318.xlsx'), sheet_name=None)
            elif 'test' in filename:
                all_sheets = pd.read_excel(os.path.join(self.dir, 'data_final_fft_test_0318.xlsx'), sheet_name=None)
            else:
                all_sheets = pd.read_excel(os.path.join(self.dir, self.filename), sheet_name=None)
        except Exception as e:
            raise e
        self.dataFrame = pd.DataFrame()
        self.sheet_names = []
        for sheet_idx, (sheet_name, df) in enumerate(all_sheets.items()):
            self.sheet_names.append(sheet_name)
            df['sheet_idx'] = sheet_idx
            self.dataFrame = pd.concat([self.dataFrame, df], ignore_index=True)
        
        self.le = []
        keys = self.dataFrame.keys()
        if self.use_type:
            le = LabelEncoder()
            self.dataFrame[keys[0]] = le.fit_transform(self.dataFrame[keys[0]])
            # print(le.fit_transform(['AD02','BYD HTH','E0Y-3Z3M','GEM','H37','H97D','KKL','M1E','MAR2 2Z','MAR2 EVA2','MBQ','MEB','NU2','SA5H','SRH','T1X','T2X RHD','X03']))
            self.le = le.classes_

    def __len__(self):
        return self.dataFrame.__len__()

    def __getitem__(self, idx):
        keys = self.dataFrame.keys()
        input = [self.dataFrame[keys[1]][idx], self.dataFrame[keys[2]][idx], self.dataFrame[keys[3]][idx], self.dataFrame[keys[5]][idx], self.dataFrame[keys[6]][idx], self.dataFrame[keys[7]][idx]]
        M1 = [self.dataFrame[keys[4]][idx]]
        # output = self.dataFrame.iloc[idx, 9:].tolist()   # 0~25600 401维
        output = self.dataFrame.iloc[idx, 9:35].tolist()   # 31.5~10000 26维
        sheet_idx = self.dataFrame['sheet_idx'][idx]

        if self.use_bowl:
            bowl = torch.tensor([self.dataFrame[keys[4]][idx]], dtype=torch.long)
        else:
            bowl = torch.tensor([0], dtype=torch.long)

        if self.transform is not None:
            input = self.transform(input)

        input = torch.tensor(input, dtype=torch.float32).clone().detach()
        output = torch.tensor(output, dtype=torch.float32).clone().detach()

        if self.use_type:
            type_ = torch.LongTensor([self.dataFrame[keys[0]][idx]])
            return input, output, type_, bowl, sheet_idx
        else:
            return input, output, bowl, sheet_idx