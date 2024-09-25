import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np

class ISCDataset(Dataset):
    def __init__(self, metadata, data_path, transform=None, ablation = None):
        self.metadata = metadata
        self.data_path = data_path
        self.transform = transform
        if transform == 'normalize':
            self.meanFeatures_perparticipant =  pd.read_csv(os.path.join(self.data_path, 'meanFeatures_perparticipant.csv'),header=None).values
            self.participantIDs = pd.read_csv(os.path.join(self.data_path, 'participantIDs.csv'),header=None).values.squeeze()
        
        self.ablation = ablation
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        data_filename = os.path.join(self.data_path, row['datafilename'])
        target_ISC = row['target_ISC']

        # Load CSV file
        data = pd.read_csv(data_filename, header=None).values
        data = np.transpose(data)
        # data = data - np.mean(data, axis=1)[:, np.newaxis]
        # participantmzean = self.meanFeatures_perparticipant.mean(axis=1)[:, np.newaxis]
        if self.transform:
            if self.transform == 'normalize':
                participantmean = self.meanFeatures_perparticipant[:,self.participantIDs == row['parID']]
                data = (data - participantmean)
            if self.transform == 'unsqueeze':
                data = data.reshape(-1,1)
                
        if self.ablation:
            list_blendshapes = ['neutral', 'browdownleft', 'browdownright', 'browinnerup', 'browouterupleft', 'browouterupright', 'cheekpuff', 'cheeksquintleft', 'cheeksquintright', 'eyeblinkleft', 'eyeblinkright', 'eyelookdownleft', 'eyelookdownright', 'eyelookinleft', 'eyelookinright', 'eyelookoutleft', 'eyelookoutright', 'eyelookupleft', 'eyelookupright', 'eyesquintleft', 'eyesquintright', 'eyewideleft', 'eyewideright', 'jawforward', 'jawleft', 'jawopen', 'jawright', 'mouthclose', 'mouthdimpleleft', 'mouthdimpleright', 'mouthfrownleft', 'mouthfrownright', 'mouthfunnel', 'mouthleft', 'mouthlowerdownleft', 'mouthlowerdownright', 'mouthpressleft', 'mouthpressright', 'mouthpucker', 'mouthright', 'mouthrolllower', 'mouthrollupper', 'mouthshruglower', 'mouthshrugupper', 'mouthsmileleft', 'mouthsmileright', 'mouthstretchleft', 'mouthstretchright', 'mouthupperupleft', 'mouthupperupright', 'nosesneerleft', 'nosesneerright']
            match self.ablation:
                case 'blendshapes':
                    data[0:52,:] = np.mean(data[0:52,:], axis=1)[:, np.newaxis]
                case 'rotation':
                    data[52:,:] = np.mean(data[52:,:], axis=1)[:, np.newaxis]
                case _:
                    # get the index on which the ablation text is contained in each of the terms in the list of blendshapes
                    idx = [i for i, bs in enumerate(list_blendshapes) if self.ablation in bs]
                    data[idx,:] = np.mean(data[idx,:], axis=1)[:, np.newaxis]

        # Convert data to torch tensors
        data = torch.tensor(data, dtype=torch.float32)
        target = torch.tensor(target_ISC, dtype=torch.float32).squeeze()

        return data, target
    
def create_dataloader_ISC(metadata, data_path, batch_size=32, shuffle=True, num_workers=0, transform=None, pin_memory=False, seed = 42):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_metadata          = metadata[metadata['split'] == 'train']
    val_metadata            = metadata[metadata['split'] == 'val']
    test_metadata           = metadata[(metadata['split'] == 'test') & (metadata['view_no'] == 1)]
    test_trainsubs_metadata = metadata[metadata['split'] == 'test_trainingsubjects']

    train_dataset          = ISCDataset(train_metadata, data_path, transform=transform)
    val_dataset            = ISCDataset(val_metadata  , data_path, transform=transform)
    test_dataset           = ISCDataset(test_metadata , data_path, transform=transform)
    test_trainsubs_dataset = ISCDataset(test_trainsubs_metadata, data_path, transform=transform)
    
    if len(train_dataset) == 0:
        train_loader = []
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    if len(val_dataset) == 0:
        val_loader = []
    else:
        val_loader   = DataLoader(val_dataset                     , batch_size=batch_size, shuffle=False   , num_workers=num_workers, pin_memory=pin_memory)

    if len(test_dataset) == 0:
        test_loader = []
    else:
        test_loader  = DataLoader(test_dataset                    , batch_size=batch_size, shuffle=False   , num_workers=num_workers, pin_memory=pin_memory)

    if len(test_trainsubs_dataset) == 0:
        test_trainsubs_loader = []
    else:
       test_trainsubs_loader = DataLoader(test_trainsubs_dataset , batch_size=batch_size, shuffle=False   , num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader, test_trainsubs_loader

def create_dataloader_ISC_alldata(metadata, data_path, batch_size=32, shuffle=True, num_workers=0, transform=None, pin_memory=False, seed = 42):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    alldata_metadata       = metadata
    alldata_dataset        = ISCDataset(alldata_metadata, data_path, transform=transform)
    
    alldata_loader        = DataLoader(alldata_dataset                 , batch_size=batch_size, shuffle=shuffle , num_workers=num_workers, pin_memory=pin_memory)
    return alldata_loader

def create_dataloader_ISC_alldata_ablation(metadata, data_path, batch_size=32, shuffle=True, num_workers=0, transform=None, pin_memory=False, seed = 42, ablation = None):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    alldata_metadata       = metadata
    alldata_dataset        = ISCDataset(alldata_metadata, data_path, transform=transform, ablation = ablation)
    
    alldata_loader_ablation        = DataLoader(alldata_dataset                 , batch_size=batch_size, shuffle=shuffle , num_workers=num_workers, pin_memory=pin_memory)
    return alldata_loader_ablation