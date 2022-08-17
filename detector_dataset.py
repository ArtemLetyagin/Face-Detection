import cv2
from torch.utils.data import Dataset
import torchvision
import torch
import pandas as pd
import numpy as np
class TrainDataset(Dataset):
    def __init__(self, df_path, path_to_train_images):
        super().__init__()
        self.path_to_train_images = path_to_train_images
        self.train_df = pd.read_csv(df_path)
        self.image_ids = self.train_df['id'].unique()
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        bboxes = self.train_df[self.train_df['id']==image_id]
        
        image = cv2.imread(self.path_to_train_images+'//'+ f'{self.image_ids[index]}'+'.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        boxes = bboxes[['xmin', 'ymin', 'xmax', 'ymax']].values

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        
        image = torchvision.transforms.ToTensor()(image)
        return image, target
    
    def __len__(self):
        return self.image_ids.shape[0]