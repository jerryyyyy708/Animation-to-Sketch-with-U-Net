import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def calculate_resize_image(shape, limit_size):
    mx = max(shape[0], shape[1]) 
    mn = min(shape[0], shape[1])
    if mx <= limit_size:
        return shape
    
    mn = int(limit_size / mx * mn)
    mx = limit_size
    return [mx, mn] if shape[0] >= shape[1] else [mn, mx]
    

class SketchDataset(Dataset):
    def __init__(self, data_path, sketch_path, limit_size):
        self.data_path = data_path
        self.sketch_path = sketch_path
        self.limit_size = limit_size

        if self.sketch_path != "":
            data_files = set(os.listdir(self.data_path))
            sketch_files = set(os.listdir(self.sketch_path))
            self.files = list(data_files & sketch_files)
        else:
            self.files = os.listdir(self.data_path)

    def __getitem__(self, index):
        filename = self.files[index]
        data_file = Image.open(self.data_path + '/' + filename)

        if self.sketch_path != "":
            sketch_file = Image.open(self.sketch_path + '/' + filename).convert('L')
            if data_file.size[:2] != sketch_file.size[:2]:
                raise Exception('The file shape is different.', filename)

        size = calculate_resize_image(data_file.size, self.limit_size)
        
        for i in range(len(size)):
            size[i] = size[i] // 16 * 16
        
        data_file = data_file.resize(size)

        to_tensor = transforms.ToTensor()
        if self.sketch_path != "":
            sketch_file = sketch_file.resize(size)
            #angle = random.random() * 90
            #data_file = data_file.rotate(angle, fillcolor="white")
            #sketch_file.rotate(angle, fillcolor="white")
            return to_tensor(data_file), to_tensor(sketch_file)
        else:
            return to_tensor(data_file)
            
    def __len__(self):
        return len(self.files)
