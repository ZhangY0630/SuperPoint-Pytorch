import os
import glob
from copy import deepcopy
from utils.params import dict_update
from dataset.utils.homographic_augmentation import homographic_aug_pipline
from dataset.utils.photometric_augmentation import PhotoAugmentor
from utils.keypoint_op import compute_keypoint_map
from dataset.utils.photometric_augmentation import *
from torchvision import transforms
from torch.utils.data import DataLoader
import random
class SelfDataset(torch.utils.data.Dataset):

    def __init__(self, config, is_train, device='cpu'):

        super(SelfDataset, self).__init__()
        self.device = device
        self.is_train = is_train
        self.resize = tuple(config['resize'])
        self.photo_augmentor = PhotoAugmentor(config['augmentation']['photometric'])
        self.config = config
        # self.valid_index = []
        if self.is_train:
            self.samples = self._init_data(config['image_train_path'], config['label_train_path'], config['pairs_train_path'])
        else:
            self.samples = self._init_data(config['image_test_path'], config['label_test_path'], config['pairs_test_path'])


    def _init_data(self, image_path, label_path=None, pair_path=None):
        ##
        if not isinstance(image_path,list):
            image_paths, label_paths, pair_paths = [image_path,], [label_path,], [pair_path,]
        else:
            image_paths, label_paths, pair_paths = image_path, label_path, pair_path

        image_types = ['jpg','jpeg','bmp','png']
        samples = []
        for im_path, lb_path, pair_path in zip(image_paths, label_paths, pair_paths):
            pairs = np.load(os.path.join(pair_path, 'pairs.npy'), allow_pickle=True)
            pairs = pairs.item()
            for idx,key in enumerate(pairs):
                
                if "test" in label_paths[0]:
                    if idx > 50:
                        break
                
                keyname = key.split(".")[0]
                temp_im = os.path.join(im_path, key)
                if lb_path is not None:
                    temp_lb = os.path.join(lb_path, keyname+'.npy')
                else:
                    temp_lb = None
                for i in range(len(pairs[key]['pairs'])):
                    pair = pairs[key]['pairs'][i]
                    pairname = pair.split(".")[0]
                    covisibility = pairs[key]['covisibility'][i]
                    index = pairs[key]['index'][i]
                    temp_im1 = os.path.join(im_path, pair)
                    if lb_path is not None:
                        temp_lb1 = os.path.join(lb_path, pairname+'.npy')
                    else:
                        temp_lb1 = None
                    # if len(index) <= 1500 and len(index) >= 1000:
                    if len(index) >= 350:
                        # temp = list(zip(index, covisibility))  # make pairs out of the two lists
                        # temp = random.sample(temp, 350)  # pick 350 random pairs
                        # index, covisibility = zip(*temp)  # separate the pairs
                        samples.append({'image':temp_im, 'label':temp_lb, 'image1':temp_im1, 'label1': temp_lb1, 'index': index, 'covisibility': covisibility})
        print(f"Num of samples: {len(samples)}")
        # num_of_pairs = len(samples[0]['index'])
        # print("The first sample's number of matches: ", num_of_pairs)
        return samples

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        '''load raw data'''
        
        data_path = self.samples[idx]
        
         # load images
        img = cv2.imread(data_path['image'], 0)#Gray image
        img = cv2.resize(img, self.resize[::-1])
        img_tensor = torch.as_tensor(img.copy(), dtype=torch.float, device=self.device)
        img1 = cv2.imread(data_path['image1'], 0)#Gray image
        img1 = cv2.resize(img1, self.resize[::-1])
        img_tensor1 = torch.as_tensor(img1.copy(), dtype=torch.float, device=self.device)

        pts = None if data_path['label'] is None else np.load(data_path['label'])[:, [1,0,2]]
        pts1 = None if data_path['label1'] is None else np.load(data_path['label1'])[:, [1,0,2]]
        
        pts[:, 0] = (pts[:, 0]+0.5)/1200*self.resize[0]
        pts[:, 1] = (pts[:, 1]+0.5)/1920*self.resize[1]
        pts1[:, 0] = (pts1[:, 0]+0.5)/1200*self.resize[0]
        pts1[:, 1] = (pts1[:, 1]+0.5)/1920*self.resize[1]
        
        kpts_tensor = None if pts is None else torch.as_tensor(pts, dtype=torch.float, device=self.device)
        kpts_tensor1 = None if pts1 is None else torch.as_tensor(pts1, dtype=torch.float, device=self.device)

        data = {   'image':{'img': img_tensor,
                             'kpts': kpts_tensor,
                             'kpts_map': None,
                             'mask': None},
                    'image1':{'img': img_tensor1,
                              'kpts': kpts_tensor1,
                              'kpts_map': None,
                              'mask': None}, 
                    'pairs': None      
        }
        # compute warpings        
        if self.is_train:
            photo_enable = self.config['augmentation']['photometric']['train_enable']
            homo_enable = self.config['augmentation']['homographic']['train_enable']
        else:
            photo_enable = self.config['augmentation']['photometric']['test_enable']
            homo_enable = self.config['augmentation']['homographic']['test_enable']

        if homo_enable and data['image']['kpts'] is not None and data['image1']['kpts'] is not None:#homographic augmentation
            data_homo = homographic_aug_pipline(data['image']['img'].to(self.device),
                                                data['image']['kpts'].to(self.device),
                                                self.config['augmentation']['homographic'],
                                                device=self.device, id_included=True)
            data_homo1 = homographic_aug_pipline(data['image1']['img'].to(self.device),
                                                data['image1']['kpts'].to(self.device),
                                                self.config['augmentation']['homographic'],
                                                device=self.device, id_included=True)
            data['image'].update(data_homo)
            data['image1'].update(data_homo1)
            
        if photo_enable:
            photo_img = data['image']['img'].cpu().numpy().round().astype(np.uint8)
            photo_img = self.photo_augmentor(photo_img)
            data['image']['img'] = torch.as_tensor(photo_img, dtype=torch.float,device=self.device)
            photo_img1 = data['image1']['img'].cpu().numpy().round().astype(np.uint8)
            photo_img1 = self.photo_augmentor(photo_img1)
            data['image1']['img'] = torch.as_tensor(photo_img1, dtype=torch.float,device=self.device)
        
        # compute new pairs and index
        pairs_dict = dict.fromkeys(data_path['covisibility'], -1)
        index_dict = dict.fromkeys(data_path['index'], -1)
        pairs_list = data_path['covisibility']
        index_list = data_path['index']
        pairs = []
        
        for i, x in enumerate(data['image']['kpts']):
            if int(x[2]) in index_dict:
                index_dict[int(x[2])] = i
        for i, x in enumerate(data['image1']['kpts']):
            if int(x[2]) in pairs_dict:
                pairs_dict[int(x[2])] = i
        
        for i in range(len(pairs_list)):
            if pairs_dict[pairs_list[i]] != -1 and index_dict[index_list[i]] != -1:
                pairs.append([index_dict[index_list[i]], pairs_dict[pairs_list[i]]])
                
        if len(pairs)>200:
            pairs = random.sample(pairs, 200)
            data['pairs'] = torch.as_tensor(np.array(pairs).astype(np.int), device=self.device)
        else:
            data['pairs'] == None
        
        # remove the old index from points & normalize images
        for image_flag in ['image','image1']:
            data[image_flag]['kpts'] = data[image_flag]['kpts'][:,:2].int()
            data[image_flag]['img'] = data[image_flag]['img']/255.
        
        return data
    

    def batch_collator(self, samples):
        """
        :param samples:a list, each element is a dict with keys
        like `img`, `img_name`, `kpts`, `kpts_map`,
        `valid_mask`, `homography`...
        img:H*W, kpts:N*2, kpts_map:HW, valid_mask:HW, homography:HW
        :return:
        """
        batch={'image' : {  'img':      [],
                            'kpts':     [],
                            'kpts_map': [],
                            'mask':     []},
               'image1' : {  'img':      [],
                            'kpts':     [],
                            'kpts_map': [],
                            'mask':     []},
               'pairs': []
               }
        for s in samples:
            
            batch['pairs'].append(s['pairs'])
            batch['image']['img'].append(s['image']['img'].unsqueeze(dim=0))
            batch['image']['kpts'].append(s['image']['kpts'])
            batch['image']['kpts_map'].append(s['image']['kpts_map'])
            batch['image']['mask'].append(s['image']['mask'])
            batch['image1']['img'].append(s['image1']['img'].unsqueeze(dim=0))
            batch['image1']['kpts'].append(s['image1']['kpts'])
            batch['image1']['kpts_map'].append(s['image1']['kpts_map'])
            batch['image1']['mask'].append(s['image1']['mask'])
        
        for k0 in ('image','image1'):
            for k1 in ('img', 'kpts_map', 'mask'):
                batch[k0][k1] = torch.stack(batch[k0][k1])
                
        torch.cuda.empty_cache()
        return batch
    
if __name__=='__main__':
    import yaml
    from dataset.utils.photometric_augmentation import *
    with open('/Users/zhouchang/Documents/GitHub/SuperPoint-Pytorch/config/superpoint_train.yaml','r') as fin:
        config = yaml.safe_load(fin)

    selfdata = SelfDataset(config['data'],True)
    cdataloader = DataLoader(selfdata,collate_fn=selfdata.batch_collator,batch_size=1,shuffle=True)

    for i,d in enumerate(cdataloader):
        if i>=5:
            break
        print(i,d)