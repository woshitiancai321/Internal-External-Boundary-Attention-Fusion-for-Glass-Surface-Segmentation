import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data

import logging
from config import cfg

num_classes = 3
ignore_label = 255
root = cfg.DATASET.TRANS10K_DIR

label2trainid = {0:0, 1:1, 2:2}
id2cat = {0: 'background', 1: 'things', 2:'stuff'}

palette = [0, 0, 0, 255, 0, 0, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.int8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def make_dataset(quality, mode):
    all_tokens = []

    assert quality == 'semantic'
    assert mode in ['train', 'validation', 'test', 'test_easy', 'test_hard']

    image_path = osp.join(root, mode, 'images')
    mask_path = osp.join(root, mode, 'masks')

    c_tokens = os.listdir(image_path)
    c_tokens.sort()
    mask_tokens = [c_token.replace('.jpg', '_mask.png') for c_token in c_tokens]

    for img_token, mask_token in zip(c_tokens, mask_tokens):
        token = (osp.join(image_path, img_token), osp.join(mask_path, mask_token))
        all_tokens.append(token)
    logging.info(f'Trans10k has a total of {len(all_tokens)} images in {mode} phase')

    logging.info(f'Trains10k-{mode}: {len(all_tokens)} images')

    return all_tokens


class Trains10kDataset(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform=target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_title = class_uniform_title
        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map

        self.data_tokens = make_dataset(quality, mode)
        self.thicky = thicky

        assert len(self.data_tokens), 'Found 0 images please check the dataset'

    def __getitem__(self, index):

        token = self.data_tokens[index]
        image_path, mask_path = token

        image, mask = Image.open(image_path).convert('RGB'), Image.open(mask_path)
        #print(mask.size)
        image_name = osp.splitext(osp.basename(image_path))[0]

        #mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
            #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        tt=mask.size[0]
        tt1=mask.size[1]
        #print(mask.size)
        mask = np.array(mask)[:, :, :3].mean(-1)
        #mask = np.transpose(mask)
        #mask1=mask
        mask[mask == 85.0] = 1
        mask[mask == 255.0] = 2
        #print(mask.max())
        assert mask.max() <= 2, mask.max()
        if((image.size[1] == tt)&(tt1 != tt)):
            mask = mask.T
        mask = Image.fromarray(mask.astype('uint8'))
        

        if self.joint_transform_list is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                image, mask = xform(image, mask)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.edge_map:
            boundary = self.get_boundary(mask, thicky=self.thicky)
            body = self.get_body(mask, boundary)
            #mask_fully=self.get_edgeAttention(mask, boundary)#hands
            attention_fully,mask_in,mask_ex=self.get_edgeAttention_diverce1(mask, boundary)#hands

            return image, mask, body, boundary, image_name ,attention_fully,mask_in,mask_ex

        return image, mask, image_name

    def __len__(self):
        return len(self.data_tokens)

    def build_epoch(self):
        pass

    @staticmethod
    def get_boundary(mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        #boundary1 = np.zeros_like(tmp)
        #boundary2=cv2.copyTo(boundary1)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        #boundary_inter = cv2.drawContours(boundary1, contour, -1, 1, -1)
        #cv2.imshow("123",mask)
        #cv2.waitKey()
        boundary = boundary.astype(np.float)
        return boundary#,boundary_inter

    @staticmethod
    def get_body(mask, edge):
        # mask=mask.numpy()
        # cv2.imshow("123",mask)
        # cv2.waitKey()
        edge_valid = edge == 1
        body = mask.clone()
        body[edge_valid] = ignore_label
        #print(mask)
        return body
    @staticmethod
    def get_edgeAttention(mask, edgeattention):#all out+int
        tmp = mask.data.numpy().astype('uint8')
        #contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        dilate=cv2.dilate(tmp,kernel)
        erode=cv2.erode(tmp,kernel1)
        boundary=(dilate-erode)
        gaussian=boundary*255;
        gaussian=cv2.GaussianBlur(gaussian,(5,5),0)

        #gaussian = gaussian.astype(np.float)
        #gaussian=gaussian/255+1
        #tmp=tmp*255
        #cv2.imshow("123",gaussian)
        #cv2.waitKey()
        #cv2.imshow("123",tmp)
        #cv2.waitKey()
        #gaussian=cv2.subtract(gaussian,tmp*255)
        #cv2.imshow("123",gaussian)
        #cv2.waitKey()
        return gaussian

    @staticmethod
    def get_edgeAttention_diverce(mask, edgeattention):#all out+int
        tmp = mask.data.numpy().astype('uint8')
        #edgeattention=edgeattention.data.numpy().astype('uint8')
        tmp[tmp>1]=1
        
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary222 = np.zeros_like(tmp)
        mask_in= np.zeros_like(tmp)
        mask_ex= np.zeros_like(tmp)
        boundary222 = cv2.drawContours(boundary222, contour, -1, 1, 3)
        boundary333 = cv2.drawContours(boundary222, contour, -1, 1, 8)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        dilate=cv2.dilate(tmp,kernel)
        erode=cv2.erode(tmp,kernel1)
        boundary=(dilate-erode)
        gaussian=boundary*255;
        gaussian=cv2.GaussianBlur(gaussian,(9,9),0)#attention
        
        interior=cv2.subtract(tmp,erode)
        external=cv2.subtract(dilate,tmp) 
        bound_in=cv2.bitwise_or(interior, boundary222)
        bound_out=cv2.bitwise_or(external, boundary222)
        #bound_in=bound_in*255
        #bound_out=bound_out*255
        #cv2.imshow("123",bound_in)
        #cv2.waitKey()
        #g=np.where(interior==1)
        #erode[erode>0]=1
        mask_ex[bound_out==1]=1
        mask_in[bound_in==1]=1#atten interior and external mask
        #tmp[external==1]=2
        #tmp=tmp*122;
        #gaussian = gaussian.astype(np.float)
        #gaussian=gaussian/255+1
        #interior=interior*255
        #external=external*255
        #cv2.imshow("123",mask_ex*255)
        #cv2.waitKey()
        #cv2.imshow("123",tmp)
        #cv2.waitKey()
        #gaussian=cv2.subtract(gaussian,tmp*255)
        #cv2.imshow("123",boundary*124)
        #cv2.waitKey()
        ttt=cv2.add(mask_ex*80,boundary333*160)
        cv2.imshow("123",ttt)
        cv2.waitKey()
        mask_ex = mask_ex.astype(int)
        mask_in= mask_in.astype(int)
        return gaussian,mask_in,mask_ex
    @staticmethod
    def get_edgeAttention(mask, edgeattention):#all out+int
        tmp = mask.data.numpy().astype('uint8')
        #contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        dilate=cv2.dilate(tmp,kernel)
        erode=cv2.erode(tmp,kernel1)
        boundary=(dilate-erode)
        gaussian=boundary*255;
        gaussian=cv2.GaussianBlur(gaussian,(5,5),0)

        #gaussian = gaussian.astype(np.float)
        #gaussian=gaussian/255+1
        #tmp=tmp*255
        #cv2.imshow("123",gaussian)
        #cv2.waitKey()
        #cv2.imshow("123",tmp)
        #cv2.waitKey()
        #gaussian=cv2.subtract(gaussian,tmp*255)
        #cv2.imshow("123",gaussian)
        #cv2.waitKey()
        return gaussian

    @staticmethod
    def get_edgeAttention_diverce1(mask, edgeattention):#all out+int
        #drawcounter is fisrt plus out thne plus inner
        tmp = mask.data.numpy().astype('uint8')
        #edgeattention=edgeattention.data.numpy().astype('uint8')
        tmp[tmp>1]=1
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary_zero1 = np.zeros_like(tmp)
        boundary_zero2 = np.zeros_like(tmp)
        mask_in= np.zeros_like(tmp)
        mask_ex= np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary_zero1, contour, -1, 1, 1)
        boundary_thickness = cv2.drawContours(boundary_zero2, contour, -1, 1, 8)
        exterior=cv2.subtract(boundary_thickness,tmp)
        exterior_bound=cv2.bitwise_or(exterior, boundary)
        interior=cv2.subtract(boundary_thickness,exterior)
        interior_bound=cv2.bitwise_or(interior, boundary)
        # interior_bound=interior_bound+exterior_bound

       
        gaussian=boundary_thickness*255;
        gaussian=cv2.GaussianBlur(gaussian,(7,7),0)#attention
        #cv2.imshow("123",gaussian)
        #cv2.waitKey()
        
        #bound_in=bound_in*255
        #bound_out=bound_out*255
        #cv2.imshow("123",bound_in)
        #cv2.waitKey()
        #g=np.where(interior==1)
        #erode[erode>0]=1
        mask_ex[exterior_bound==1]=1
        mask_in[interior_bound==1]=1#atten interior and external mask
        #tmp[external==1]=2
        #tmp=tmp*122;
        #gaussian = gaussian.astype(np.float)
        #gaussian=gaussian/255+1
        #interior=interior*255
        #external=external*255
        #cv2.imshow("123",mask_ex*255)
        #cv2.waitKey()
        #cv2.imshow("123",tmp)
        #cv2.waitKey()
        #gaussian=cv2.subtract(gaussian,tmp*255)
        #cv2.imshow("123",boundary*124)
        #cv2.waitKey()
        #ttt=cv2.add(mask_ex*80,boundary333*160)
        #cv2.imshow("123",mask_ex*255)
        #cv2.waitKey()
        mask_ex = mask_ex.astype(int)
        mask_in= mask_in.astype(int)
        return gaussian,mask_in,mask_ex