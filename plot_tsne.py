import models
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import csv
import os

class tsne_dataset(data.Dataset):
    def __init__(self, dir, name, class_list,img_size):
        self.dir = dir
        self.name = name
        self.image_size = img_size
        self.class_list = class_list
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        ])
        file = os.path.join(self.dir,self.name,'{}_test.csv'.format(self.name))
        self.fnames = []
        self.labels = []
        with open(file) as f:
            csv_reader = csv.reader(f,delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    for class_label in class_list:
                        if int(row[1]) == int(class_label):
                            self.fnames.append(row[0])
                            self.labels.append(int(row[1]))
                    line_count += 1
                else:
                    line_count += 1
        
        self.num_samples = len(self.fnames)
        print(self.num_samples,len(self.labels))

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        l = self.labels[idx]
        img = Image.open(os.path.join(self.dir + fname)).convert('RGB')
        img = self.transform(img)

        return img, l   

    def __len__(self):
        return self.num_samples     

def main():
    batch_size = 32

    root = 'data/'
    path = 'checkpoints/sketch-30.pth'
    output_path = './visual/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    source1 = 'infograph'
    source2 = 'quickdraw'
    source3 = 'real'
    target = 'sketch'

    feature_extractor = models.feature_extractor()

    state = torch.load(path)

    feature_extractor.load_state_dict(state['feature_extractor'])

    # Get image list
    #class_list = [0,1,2,3,4,5,6,7,8,9,10]
    class_list = [0]
    dataset_s1 = tsne_dataset(dir = root,name = source1,class_list=class_list,img_size=(224,224))
    dataset_s2 = tsne_dataset(dir = root,name = source2,class_list=class_list,img_size=(224,224))
    dataset_t = tsne_dataset(dir = root,name = target,class_list=class_list,img_size=(224,224))

    dataloader_s1 = DataLoader(dataset_s1, batch_size=batch_size, shuffle=False,num_workers=4)
    dataloader_s2 = DataLoader(dataset_s2, batch_size=batch_size, shuffle=False,num_workers=4)
    dataloader_t = DataLoader(dataset_t, batch_size=batch_size, shuffle=False,num_workers=4)

    if torch.cuda.is_available():
        feature_extractor = feature_extractor.cuda()

    feature_extractor.eval()
    feature_list_s1 = torch.zeros((1,2048))
    feature_list_s2 = torch.zeros((1,2048))
    feature_list_t = torch.zeros((1,2048))
    label_list_s1 = torch.zeros(1).long()
    label_list_s2 = torch.zeros(1).long()
    label_list_t = torch.zeros(1).long()

    with torch.no_grad():
        for idx, (data_1, data_2, data_t) in enumerate(zip(dataloader_s1,dataloader_s2,dataloader_t)):
            img1, lb1 = data_1
            img2, lb2 = data_2
            imgt, lbt = data_t
            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()
                imgt = imgt.cuda()
                #a = a.cuda()
            
            ft_1 = feature_extractor(img1)
            ft_2 = feature_extractor(img2)
            ft_t = feature_extractor(imgt)
            feature_list_s1 = torch.cat((feature_list_s1,ft_1))
            feature_list_s2 = torch.cat((feature_list_s2,ft_1))
            feature_list_t = torch.cat((feature_list_t,ft_1))
            b = torch.cat((b,lb1))
            if idx == 0:
                feature_list_s1 = feature_list_s1[1:,:]
                feature_list_s2 = feature_list_s2[1:,:]
                feature_list_t = feature_list_t[1:,:]
                b = b[1:]
            print(a.size(),ft_1.size(),b.size(),img1.size())
            feature_list_s1.append(ft_1.cpu().data.numpy())
            feature_list_s2.append(ft_2.cpu())
            feature_list_t.append(ft_t.cpu())

            label_list_s1.append(lb1)
            label_list_s2.append(lb2)
            label_list_t.append(lbt)
    #print(feature_list_s1[0])
    tsne_ft1 = TSNE(n_components=2,perplexity=40.0,random_state=24,verbose=1).fit_transform(a.cpu().data.numpy())

    cm = plt.cm.get_cmap("jet",10)
    plt.figure(figsize=(8,6))
    plt.title('tSNE feature1')
    plt.scatter(tsne_ft1[:,0],tsne_ft1[:,1],c=b.cpu().data.numpy(),cmap=cm)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5,9.5)
    plt.savefig('{}{}_tSNE.jpg'.format(output_path,target))


if __name__ == '__main__':
    main()
