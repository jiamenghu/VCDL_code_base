import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from utils import *
from prefetch_generator import BackgroundGenerator
import threading
import queue as Queue

#训练验证集数据类
class DataLoaderTrain(Dataset):
    def __init__(self,train_opts):

        super().__init__()
        self.sample_list=train_opts['seq_list']
        self.data_aug=train_opts['data_aug']
        self.data_aug_rate=train_opts['data_aug_rate']
        self.patch_size=train_opts['patch_size']
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,index):

        sample=self.read_sample(self.sample_list[index])

        return sample

    def read_sample(self,elem_dir): 

        if self.patch_size==-1:
            reco_Y=read_frame_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
                        elem_dir['consec_fra'])
            high_Y=read_frame_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'],)            

        else:
            
            top = random.randint(0, elem_dir['height'] - self.patch_size)
            left = random.randint(0, elem_dir['width'] - self.patch_size)

            reco_Y=read_patch_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
                                    elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
                                    [left,top],self.patch_size,elem_dir['consec_fra'])

            high_Y=read_patch_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
                                    elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'],
                                    [left,top],self.patch_size,)
        
        
        img_aug=[reco_Y,high_Y]

        if self.data_aug:
            img_aug= augment(img_aug,self.data_aug_rate)


        reco_Y=np.array(img_aug[0],dtype=np.float32)
        high_Y=np.array(img_aug[1],dtype=np.float32)

        #归一化
        reco_Y=normalize_zero(reco_Y,elem_dir['BitDepth']['Reco'])
        high_Y=normalize_zero(high_Y,elem_dir['BitDepth']['High'])        

        sample={'Reco_Y':reco_Y,'High_Y':high_Y}

        # if 'Extra_channel' in elem_dir:
        #     extra_Y=read_Y_multipatchdata(elem_dir['Extra_channel'],elem_dir['width'],elem_dir['height'],
        #                             elem_dir['BitDepth']['Extra_channel'],elem_dir['frame_index'],
        #                             elem_dir['patch_pos'],[elem_dir['patch_size'],elem_dir['patch_size']],self.consec_frames)
        #     if extra_Y.max()!=0:
        #         extra_Y/=extra_Y.max()
        #     #extra_Y=normalize_zero(extra_Y,low_BitDepth)    
        #     sample['extra_Y']  = extra_Y


        return sample

class DataLoaderValid(Dataset):
    def __init__(self,valid_opts):

        super().__init__()
        self.sample_list=valid_opts['seq_list']
        self.data_aug=valid_opts['data_aug']
        self.patch_size=valid_opts['patch_size']

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,index):
        sample=self.read_sample(self.sample_list[index])
        return sample

    def read_sample(self,elem_dir): 

        if self.patch_size==-1:
            reco_Y=read_frame_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
                        elem_dir['consec_fra'])
            high_Y=read_frame_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'],)            

        else:
            
            top = random.randint(0, elem_dir['height'] - self.patch_size)
            left = random.randint(0, elem_dir['width'] - self.patch_size)

            reco_Y=read_patch_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
                                    elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
                                    [left,top],self.patch_size,elem_dir['consec_fra'])

            high_Y=read_patch_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
                                    elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'],
                                    [left,top],self.patch_size,)

        img_aug=[reco_Y,high_Y]
        reco_Y=np.array(img_aug[0],dtype=np.float32)
        high_Y=np.array(img_aug[1],dtype=np.float32)
        # print(elem_dir,'reco',reco_Y,'high',high_Y)
        # assert False

        #归一化

        reco_Y=normalize_zero(reco_Y,elem_dir['BitDepth']['Reco'])
        high_Y=normalize_zero(high_Y,elem_dir['BitDepth']['High'])        

        sample={'Reco_Y':reco_Y,'High_Y':high_Y}

        # if 'Extra_channel' in elem_dir:
        #     extra_Y=read_Y_multipatchdata(elem_dir['Extra_channel'],elem_dir['width'],elem_dir['height'],
        #                             elem_dir['BitDepth']['Extra_channel'],elem_dir['frame_index'],
        #                             elem_dir['patch_pos'],[elem_dir['patch_size'],elem_dir['patch_size']],self.consec_frames)
        #     if extra_Y.max()!=0:
        #         extra_Y/=extra_Y.max()
        #     #extra_Y=normalize_zero(extra_Y,low_BitDepth)    
        #     sample['extra_Y']  = extra_Y


        return sample

class DataLoaderTest(Dataset):
    def __init__(self,test_opts):
        super().__init__()
        self.sample_list=test_opts['sep_list']
        self.patch_size=test_opts['patch_size']
        self.baseline=test_opts['baseline']
        #TODO patch_size test

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self,index):
        sample=self.read_sample(self.sample_list[index])
        return sample

    def read_sample(self,elem_dir): 

        if self.patch_size==-1:
            reco_Y=read_frame_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
                        elem_dir['consec_fra'])
            high_Y=read_frame_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'])    
            if self.baseline:
                vtms_Y=read_frame_yuv(elem_dir['Vtms'],elem_dir['width'],elem_dir['height'],
                        elem_dir['BitDepth']['Vtms'],elem_dir['gt_frame_index'])    

        else:
            assert False,'TODO'

            # reco_Y=read_patch_yuv(elem_dir['Reco'],elem_dir['width'],elem_dir['height'],
            #                         elem_dir['BitDepth']['Reco'],elem_dir['lq_frame_index'],
            #                         [left,top],self.patch_size,elem_dir['consec_fra'])

            # high_Y=read_patch_yuv(elem_dir['High'],elem_dir['width'],elem_dir['height'],
            #                         elem_dir['BitDepth']['High'],elem_dir['gt_frame_index'],
            #                         [left,top],self.patch_size)
        
        img_aug=[reco_Y,high_Y]

        reco_Y=np.array(img_aug[0],dtype=np.float32)
        high_Y=np.array(img_aug[1],dtype=np.float32)
        if self.baseline : vtms_Y=np.array(img_aug[2],dtype=np.float32)
        # print(elem_dir,'reco',reco_Y,'high',high_Y)
        # assert False

        #归一化
        reco_Y=normalize_zero(reco_Y,elem_dir['BitDepth']['Reco'])
        high_Y=high_Y*2**(elem_dir['BitDepth']['High']-8)      

        sample={'Reco_Y':reco_Y,'High_Y':high_Y}
        if self.baseline : sample['Vtms_Y']=vtms_Y

        # if 'Extra_channel' in elem_dir:
        #     extra_Y=read_Y_multipatchdata(elem_dir['Extra_channel'],elem_dir['width'],elem_dir['height'],
        #                             elem_dir['BitDepth']['Extra_channel'],elem_dir['frame_index'],
        #                             elem_dir['patch_pos'],[elem_dir['patch_size'],elem_dir['patch_size']],self.consec_frames)
        #     if extra_Y.max()!=0:
        #         extra_Y/=extra_Y.max()
        #     #extra_Y=normalize_zero(extra_Y,low_BitDepth)    
        #     sample['extra_Y']  = extra_Y


        return sample


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)



class CPUPrefetcher():
    """CPU prefetcher.

    Args:
        loader: Dataloader.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()


