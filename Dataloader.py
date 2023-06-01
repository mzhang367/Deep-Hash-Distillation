from config import *

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, folder=None):
        self.img_dir = os.path.join(img_dir, folder) if folder is not None else img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.resize1 = Kg.LongestMaxSize(256)
        self.resize2 = Kg.PadTo((256, 256))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.file_list[idx][0])
        image = Image.open(img_name).convert("RGB")

        image = transforms.ToTensor()(image)
        image = self.resize1(image)
        image = self.resize2(image)
        
        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]   # here we extract each label from positions after filename
                label = T.FloatTensor(label)
            else:
                label = int(self.file_list[idx][1])
            # print(image.shape, label.shape) find an image with one channel only!
            return image[0], label
        else:
            return image[0]
        

if __name__ == '__main__':
        
        dname = 'coco'
        Image_dir = os.path.join("../data", dname)
        Train_dir = os.path.join("./data", dname+'_Train.txt')
        
        trainset = Loader(Image_dir, Train_dir, 80)
        print(trainset[36][0].shape, trainset[36][1].shape)

