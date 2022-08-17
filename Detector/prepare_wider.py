import os
import cv2
import numpy as np
import pandas as pd

class Dataset_maker():
    def __init__(self, wider, path_annot):
        self.wider = wider
        self.wider_list = os.listdir(wider)
        file = open(path_annot, 'r')
        text = file.read()
        self.txt_annot = text.split('\n')

    def create_folders(self, path_for_save, number):
        root = os.getcwd() 
        annot = []
        i=0
        for folder in self.wider_list:
            for name in os.listdir(self.wider+folder)[:number]:
                img = cv2.imread(self.wider+folder+'/'+name)
                os.chdir(path_for_save)
                cv2.imwrite(f'{i}.jpg', img)
                i+=1
                ind = self.txt_annot.index(folder+'/'+name)
                num = int(self.txt_annot[ind+1])
                coords=[]
                for j in range(num):
                    coords.append(self.txt_annot[ind+2+j].split(' ')[:4])
                annot.append(coords)
                os.chdir(root)
        
        np.save(root+'/'+'annotations_for_Faces_train_faster_rcnn.npy', annot)
        self.path_annot = root+'/'+'annotations_for_Faces_train_faster_rcnn.npy'
        print(f'{i} images saved successfully')

    def dataframe(self, name):
        self.name = name
        def convert(a):
            x = a[0]
            y = a[1]
            w = a[2]
            h = a[3]
            return [int(x), int(y), int(x)+int(w), int(y)+int(h)]

        annot_numpy_array = np.load(self.path_annot, allow_pickle=True)
        df_data = pd.DataFrame(columns=['id', 'xmin', 'ymin', 'xmax', 'ymax'])
        for i in range(len(annot_numpy_array)):
            for j in range(len(annot_numpy_array[i])):
                appender = [i]+convert(annot_numpy_array[i][j])
                df_data.loc[len(df_data.index)] = appender
        df_data.to_csv(f'{name}', index=False)
        print('DataFrame saved successfully')

    def get_dataframe(self):
        return pd.read_csv(self.name)