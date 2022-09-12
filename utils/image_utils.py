import torch
import numpy as np
#import skimage.metrics as skm  #ssim
import random
import skimage.color as skc 
from cv2 import cv2



#########计算PSNR/SSIM###########

def torchPSNR(tar_img, prd_img,lower_bound=-1,upper_bound=1):
    imdff = torch.clamp(prd_img,lower_bound,upper_bound) - torch.clamp(tar_img,lower_bound,upper_bound)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10((upper_bound-lower_bound)/rmse)
    return ps

def numpyPSNR(tar_img, prd_img, max_value=255):
    
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    if rmse==0:
        return 100
    ps = 20*np.log10(max_value/rmse)
    return ps

def calculate_ssim(img0, img1, data_range=None):
    """Calculate SSIM (Structural SIMilarity).

    Args:
        img0 (ndarray)
        img1 (ndarray)
        data_range (int, optional): Distance between minimum and maximum 
            possible values). By default, this is estimated from the image 
            data-type.
    
    Return:
        ssim (float)
    """
    ssim = skm.structural_similarity(img0, img1, data_range=data_range)
    return ssim

############数据增强#####################

def augment(imgs, aug_rate=0.5,hflip=True, rotation=True, flows=None):
    """Augment: horizontal flips or rotate (0, 90, 180, 270 degrees).

    Use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray]: Image list to be augmented.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation or not. Default: True.
        flows (list[ndarray]: Flow list to be augmented.
            Dimension is (h, w, 2). Default: None.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < aug_rate
    vflip = rotation and random.random() < aug_rate
    rot90 = rotation and random.random() < aug_rate

    def _imflip_(img, direction='horizontal'):
        """Inplace flip an image horizontally or vertically.

        Args:
            img (ndarray): Image to be flipped.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            ndarray: The flipped image (inplace).
        """
        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            for i in range(img.shape[0]): 
                cv2.flip(img[i], 1, img[i])
            return 
        elif direction == 'vertical':
            for i in range(img.shape[0]): 
                cv2.flip(img[i], 0, img[i]) 
            return 
        else:
            for i in range(img.shape[0]): 
                return cv2.flip(img[i], -1, img[i]) 

    def _augment(img):
        if hflip:
            _imflip_(img, 'horizontal')
        if vflip:
            _imflip_(img, 'vertical')
        if rot90:  # for (3,H W) image, H <-> W
            img = img.transpose(0, 2, 1)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    def _augment_flow(flow):
        if hflip:
            _imflip_(flow, 'horizontal')
            flow[:, :, 0] *= -1
        if vflip:
            _imflip_(flow, 'vertical')
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        return imgs

############数据分块与拼接#####################
class PartitionMergeBlock():
    '''
    class PartitionMergeBlock
    支持1920*1080，2560*1600，3840*2160
    
    '''

    def __init__(self,imgs,Stride=None,Scale=None):

        if not isinstance(imgs, list):
            self.imgs = [imgs]
        else:
            self.imgs=imgs 
        self.B,self.C,self.H,self.W=self.imgs[0].shape
        if not Scale:
            self.Scale=2

        else:
            self.Scale=Scale
        #print(self.B,self.C,self.H,self.W)
        if not Stride:
            if self.W==1920 and self.H==1080 :
                if self.Scale==2:
                    self.WStride=128
                    self.HStride=124
            if self.W==2560 and self.H==1600 :
                if self.Scale==2:
                    self.WStride=128
                    self.HStride=128
            if self.W==3840 and self.H==2160 :
                if self.Scale==2:
                    self.WStride=124
                    self.HStride=128
            if self.W==1280 and self.H==720 :
                if self.Scale==2:
                    self.WStride=128
                    self.HStride=128
            if self.W==2044 and self.H==1208:
                if self.Scale==2:
                    self.WStride=130
                    self.HStride=132
            if self.W==1408 and self.H==928:
                if self.Scale==2:
                    self.WStride=128
                    self.HStride=128    
            if self.W==1088 and self.H==664:
                if self.Scale==2:
                    self.WStride=128
                    self.HStride=132                             
        else:
            self.WStride=Stride[0]
            self.HStride=Stride[1]

        self.bH=self.H//self.Scale
        self.bW=self.W//self.Scale
        self.pos=self.compute_position()
        self.blcoks=self.partition()

        self.num=0
        pass

    def compute_position(self):
        assert self.Scale==2,'Not implement'
        pos=[[[0,0],[self.bH,self.bW]],
             [[0,self.bW],[self.bH,2*self.bW]],
             [[self.bH,0],[2*self.bH,self.bW]],
             [[self.bH,self.bW],[2*self.bH,2*self.bW]]
        ]

        return pos
    def partition(self):
        blcoks=[]
        for img in self.imgs:
            cur_blcok=[]
            for p in self.pos:
                x1=p[0][0] if p[0][0]==0 else p[0][0]-self.HStride
                y1=p[0][1] if p[0][1]==0 else p[0][1]-self.WStride
                x2=p[1][0] if p[1][0]==self.H else p[1][0]+self.HStride
                y2=p[1][1] if p[1][1]==self.W else p[1][1]+self.WStride
                #print(x1,y1,x2,y2)
                cur=img[:,:,x1:x2,y1:y2]
                cur_blcok.append(cur)
            blcoks.append(cur_blcok)
        return blcoks



    def merge(self,blocks=None):
        # if  isinstance(blocks[0], tensor):
        #print('merge',blocks)
        merge_img=torch.zeros_like(self.imgs[0])
        # else:
        #     merge_img=np.zeros_like(self.imgs[0])
        i=0
        if not blocks:
            blocks=self.blcoks
            #TODO 所有帧
        for p in self.pos:

            x1=0 if p[0][0]==0 else self.HStride
            y1=0 if p[0][1]==0 else self.WStride
            x2=x1+self.bH# if p[1][0]==self.H else self.bH+self.HStride
            y2=y1+self.bW #if p[1][1]==self.W else self.bW+self.WStride
            #print(merge_img.shape,blocks[i].shape)
            #print(p[0][0],p[1][0],p[0][1],p[1][1])
            #print(x1,x2,y1,y2)
            #print(merge_img[:,:,p[0][0]:p[1][0],p[0][1]:p[1][1]].shape,blocks[i][:,:,x1:x2,y1:y2].shape)
            merge_img[:,:,p[0][0]:p[1][0],p[0][1]:p[1][1]]=blocks[i][:,:,x1:x2,y1:y2]
            i+=1
        return merge_img

    def __iter__(self):
        return self
    def __next__(self):
        if self.num<len(self.blcoks[0]):
            ret=[block[self.num] for block in self.blcoks]
            self.num+=1
            return ret
        else:
            raise StopIteration

############归一化###############

def normalize(x,BitDepth):
    pixel_value=255. if BitDepth==8 else 1023.
    x = x / pixel_value
    return truncate(x, 0., 1.)
def normalize_resi(x,BitDepth=8):
    pixel_value=255. if BitDepth==8 else 1023.
    x_max=x.max()
    x_min=x.min()
    if x_max!=x_min:
        x = (x-x.min()) / (x_max-x_min)
    else:
        x = x / pixel_value
    return x
def normalize_zero(x,BitDepth=10):
    pixel_value=255. if BitDepth==8 else 1023.
    x = x*2 / pixel_value-1.0
    return truncate(x, -1., 1.)

############逆归一化##########

def denormalize(x,BitDepth=8):
    pixel_value=255. if BitDepth==8 else 1023.
    x = x * pixel_value
    return truncate(x, 0., pixel_value)
def denormalize_zero(x,BitDepth=10):
    pixel_value=255. if BitDepth==8 else 1023.
    x = (x+1.) * pixel_value/2 
    return truncate(x, 0., pixel_value)



############截断############ 

def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input  



###########类型转换###########

def img2float32(img):
    """Convert the type and range of the input image into np.float32 and [0, 1].

    Args:
        img (img in ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type, with unknown range.

    Return:
        img (ndarray): The converted image with type of np.float32 and 
        range of [0, 1].
    """
    img_type = img.dtype
    assert img_type in (np.uint8, np.float32), (
        f'The image type should be np.float32 or np.uint8, but got {img_type}')
    
    if img_type == np.uint8:  # the range must be [0, 255]
        img = img.astype(np.float32)
        img /= 255.
    else:  # np.float32, may excess the range [0, 1]
        img = img.clip(0, 1)

    return img


def ndarray2img(ndarray):
    """Convert the type and range of the input ndarray into np.uint8 and 
    [0, 255].

    Args:
        ndarray (ndarray):
            1. np.uint8 type (of course with range [0, 255]).
            2. np.float32 type with unknown range.

    Return:
        img (img in ndarray): The converted image with type of np.uint8 and 
        range of [0, 255].
    
    
    对float32类型分情况讨论: 
        1. 如果最大值超过阈值, 则视为较黑的图像, 直接clip处理；
        2. 否则, 视为[0, 1]图像处理后的结果, 乘以255.再clip.
    
    不能直接astype, 该操作会删除小数, 不精确. 应先round, 再clip, 再转换格式.
    
    image -> img2float32 -> ndarray2img 应能准确还原.
    """
    data_type = ndarray.dtype
    assert data_type in (np.uint8, np.float32), (
        f'The data type should be np.float32 or np.uint8, but got {data_type}')

    if data_type == np.float32:
        detection_threshold = 2
        if (ndarray < detection_threshold).all():  # just excess [0, 1] slightly
            ndarray *= 255.
        else:  # almost a black picture
            pass
        img = ndarray.round()  # first round. directly astype will cut decimals
        img = img.clip(0, 255)  # or, -1 -> 255, -2 -> 254!
        img = img.astype(np.uint8)
    else:
        img = ndarray

    return img


def rgb2ycbcr(rgb_img):
    """RGB to YCbCr color space conversion.

    Args:
        rgb_img (img in ndarray): (..., 3) format.

    Return:
        ycbcr_img (img in ndarray): (..., 3) format.

    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `rgb2ycbcr` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    ycbcr_img = skc.rgb2ycbcr(rgb_img)
    return ycbcr_img


def ycbcr2rgb(ycbcr_img):
    """YCbCr to RGB color space conversion.

    Args:
        ycbcr_img (img in ndarray): (..., 3) format.

    Return:
        rgb_img (img in ndarray): (..., 3) format.

    Error:
        ycbcr_img is not in (..., 3) format.

    Input image, not float array!

    Y is between 16 and 235.
    
    YCbCr image has the same dimensions as input RGB image.
    
    This function produces the same results as Matlab's `ycbcr2rgb` function.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    """
    rgb_img = skc.rgb2ycbcr(ycbcr_img)
    return rgb_img


def rgb2gray(rgb_img):
    """Compute luminance of an RGB image.

    Args:
        rgb_img (img in ndarray): (..., 3) format.

    Return:
        gray_img (single channel img in array)

    Error:
        rgb_img is not in (..., 3) format.

    Input image, not float array!

    alpha通道会被忽略.
    """
    gray_img = skc.rgb2gray(rgb_img)
    return gray_img


def gray2rgb(gray_img):
    """Create an RGB representation of a gray-level image.

    Args:
        gray_img (img in ndarray): (..., 1) or (... , ) format.

    Return:
        rgb_img (img in ndarray)
    
    Input image, not float array!

    其实还有一个alpha通道参数, 但不常用. 参见: 
    https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.gray2rgb
    """
    rgb_img = skc.gray2rgb(gray_img, alpha=None)
    return rgb_img


def bgr2rgb(img):
    code = getattr(cv2, 'COLOR_BGR2RGB')
    img = cv2.cvtColor(img, code)
    return img


def rgb2bgr(img):
    code = getattr(cv2, 'COLOR_RGB2BGR')
    img = cv2.cvtColor(img, code)
    return img
