#TODO  warmup
#TODO  log——show
#TODO   分割测试，在拼接
#TODO   测试可以选择打印每帧log。
#TODO   特征图可视化
#TODO   summory -训练完自动提取
#TODO   提示训练模型和日志文件名字不相符


from importlib.util import module_for_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
#from warmup_scheduler import GradualWarmupScheduler
#from pdb import set_trace as stx

import os
import random
import time
from datetime import datetime
import numpy as np

from config import *
from utils import *
from dataset import *
from train_test import *
from losses import *
from NN import *

############ Set Seeds #####################

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(2021)

###########setting#########################
if len(GPU)>0:
    gpus = ','.join([str(i) for i in GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    if TRAIN_BATCH_SIZE==VAL_BATCH_SIZE:
        torch.backends.cudnn.benchmark = True
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


############# Model ########################

#model_restoration=STDFNet(T=CONSEC_FRAME)
#model_restoration=STDFNet_BN(T=CONSEC_FRAME)

#model_restoration=MGO_VEN(T=CONSEC_FRAME)
#model_restoration=MGO_VEN_OpsDcn(T=CONSEC_FRAME)

#model_restoration=MGO_VEN_TP_v0(T=CONSEC_FRAME)
#model_restoration=VEN_TP_v2(T=CONSEC_FRAME)
#model_restoration=VEN_PPPN(T=CONSEC_FRAME)
#model_restoration=VEN_TP_v2_Attention(T=CONSEC_FRAME)
#model_restoration=VEN_TP_v2_Attention_compare(T=CONSEC_FRAME)


#model_restoration=STO_OTA_MGO(T=CONSEC_FRAME)
#model_restoration=STO_OTA(T=CONSEC_FRAME)  

#model_restoration=SO_MGO(T=CONSEC_FRAME)

#model_restoration=TO_MGO(T=CONSEC_FRAME)

#model_restoration=STO_MGO(T=CONSEC_FRAME)  

model_restoration=STO_OTA_v1(T=CONSEC_FRAME)

#model_restoration=STO_OTA_v1_MGO(T=CONSEC_FRAME)

#model_restoration=MGO_VEN_msO(T=CONSEC_FRAME)
#model_restoration=MGO_VEN(T=CONSEC_FRAME)

#model_restoration=STO_MGO(T=CONSEC_FRAME)
#model_restoration=SO_MGO(T=CONSEC_FRAME)


#model_restoration=DualHRNet_DCN(T=CONSEC_FRAME)
#model_restoration=DualHRNet_DCN_v1(T=CONSEC_FRAME)
#model_restoration=DualHRNet_DCN_v2(T=CONSEC_FRAME)
#model_restoration=DualHRNet_DCN_v3(T=CONSEC_FRAME)

# print(model_restoration)
# for name,param in model_restoration.named_parameters():
#     print(name)
# assert False
print(model_restoration.__class__.__name__)
total_params = sum(p.numel() for p in model_restoration.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model_restoration.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

pre_train_nn_tail_name='tail.seq.0.weight'

if not CPU and len(GPU)>0:
    model_restoration.cuda()
if len(GPU)>1:
    model_restoration = nn.DataParallel(model_restoration)#, device_ids = GPU)


######### Optimizer ############

lr= LR_INITIAL
optimizer = optim.Adam(model_restoration.parameters(), lr=lr, betas=(0.9, 0.999),eps=1e-8)

######## Loss #################
criterion = CharbonnierLoss()
# criterion_edge = losses.EdgeLoss()
#criterion = nn.MSELoss()
loss_rate={22:1e-5,27:1e-5,32:1e-5,37:1e-5}
#criterion=FrequencyMSELoss(rate=loss_rate[QP])

######## Scheduler ##############
scheduler_opts={
    'model':model_restoration,
    "optim":optimizer,
    'patience_max':PATIENCE_NUM,
    'lr':lr,
    'lr_min':LR_MIN,
    'lr_decay_rate':LR_DECAY_RATE,
    'model_path':model_path
}
scheduler_lr=MyScheduler(scheduler_opts)


# warmup_epochs = 3
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
# scheduler.step()

# if RESUME:
#     path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
#     utils.load_checkpoint(model_restoration,path_chk_rest)
#     start_epoch = utils.load_start_epoch(path_chk_rest) + 1
#     utils.load_optim(optimizer, path_chk_rest)
#     for i in range(1, start_epoch):
#         scheduler.step()
#     new_lr = scheduler.get_lr()[0]
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:", new_lr)
#     print('------------------------------------------------------------------------------')


######### DataLoaders ###########

print(datetime.now(),'start process data',flush=True)
process_data_time=time.time()

train_seq_dir_list = get_seq_dir_list(TRAIN_dir,EXTRA_CHANNEL,TRAIN=True)  #element seqencce dir 
print('seq for train:',len(train_seq_dir_list))
train_elem_dir_list = get_elem_dir_list(train_seq_dir_list ,\
     {'consec_frames':CONSEC_FRAME,'extra_channel':EXTRA_CHANNEL,'train':True,'baseline':BaseLine})  #element data_patch dir
np.random.shuffle(train_elem_dir_list)
print('elem for train:',len(train_elem_dir_list))
trainset_opts={
        'seq_list':train_elem_dir_list,
        'data_aug':DATA_AUG,
        'data_aug_rate':0.5,
        'patch_size':TRAIN_PS,
}
train_dataset = DataLoaderTrain(trainset_opts)      #Dataset实例
train_loader = DataLoaderX(dataset=train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True, pin_memory=False) #drop_last=False,


valid_seq_dir_list = get_seq_dir_list(VAL_dir,EXTRA_CHANNEL,TRAIN=True)  #element seqencce dir 
print('seq for valid:',len(valid_seq_dir_list))
valid_elem_dir_list = get_elem_dir_list(valid_seq_dir_list , \
        {'consec_frames':CONSEC_FRAME,'extra_channel':EXTRA_CHANNEL,'train':True,'baseline':BaseLine})  #element data_patch dir
print('elem for valid:',len(valid_elem_dir_list))
validset_opts={
        'seq_list':valid_elem_dir_list,
        'data_aug':False,
        'data_aug_rate':0.5,
        'patch_size':VAL_PS,
}
valid_dataset = DataLoaderValid(validset_opts)      #Dataset实例
valid_loader = DataLoaderX(dataset=valid_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4,  pin_memory=False) #drop_last=False,

print(datetime.now(),'done process data, using',(time.time()-process_data_time)/60,'m! ',flush=True)

################ Train ################################
if not TEST :
    print('&&===> Start Train !!!')

    # if not RESUME:
    #     pre_train(train_loader,model=model_restoration,pargms_name=pre_train_nn_tail_name)

    if not PRINT_BATCH:
        val_res=valid(-1,valid_loader,model_restoration,criterion,baseline=BaseLine,PartitionBlock=PartitionBlock)
        print(datetime.now(),' valid: ',val_res)
        best_psnr_gain=val_res['gain']

    epoch_start_time = time.time()
    for epoch in range(0,MAX_EPOCHS):

        train(epoch,train_loader,model_restoration,criterion,optimizer,\
            {'print_batch':PRINT_BATCH,'thr_clip':THR_CLIP})

        if epoch%VAL_Interval==0 :
            val_res=valid(epoch,valid_loader,model_restoration,criterion,baseline=BaseLine,PartitionBlock=PartitionBlock)
            cur_psnr_gain=val_res['gain']
            print(datetime.now(),' valid: ',val_res)
            scheduler_lr.step(cur_psnr_gain)
            if scheduler_lr.stop(): 
                break
        
        #scheduler.step()
    
    print(datetime.now(), 'train done ( training time:'+ str((time.time()-epoch_start_time)/3600) +'h ,average epoch time:'+str((time.time()-epoch_start_time)/60/epoch)+'m )')
    print("psnr gain:",scheduler_lr.best_psnr_gain)
    torch.save(model_restoration.state_dict(), model_path)
    print("model saved to {} ".format(model_path))


print('&& Start TEST')
test_start_time=time.time()
if TEST:
    if CPU:
        model_restoration.load_state_dict(torch.load(model_path, map_location='cpu'))  
    else:
        model_restoration.load_state_dict(torch.load(model_path)) 

# for name,parameters in model.named_parameters():
#     print(name,':',torch.norm(parameters))
# assert False,'over'
if not CPU:
    torch.backends.cudnn.benchmark = False

for Test_class in TEST_dir:

    test8_seq_dir_list = get_seq_dir_list(Test_class,EXTRA_CHANNEL,TRAIN=False,baseline=BaseLine)  #element seqencce dir 
    print('seq for test8:',len(test8_seq_dir_list))
    
    for i in range(len(test8_seq_dir_list)):
        name=test8_seq_dir_list[i]['High'].split('/')[-1]
        # if 'VTM' in test8_seq_dir_list[i]['High'] and  ONLY_B :
        #     if "MarketPlace" not in name and "RitualDance" not in name:
        #             continue
        cur_seq=test8_seq_dir_list[i]
        cur_elem_dir_list=get_elem_dir_list([cur_seq] , \
            {'patch_size':TEST_PS,'consec_frames':CONSEC_FRAME,'extra_channel':EXTRA_CHANNEL,'train':False,'baseline':BaseLine})  #element data_patch dir
        print(name,'elem :',len(cur_elem_dir_list))
        test_opts={
            'sep_list':cur_elem_dir_list,
            'patch_size':TEST_PS,
            'baseline':BaseLine,
        }
        cur_dataset = DataLoaderTest(test_opts)      #Dataset实例
        cur_loader = DataLoaderX(dataset=cur_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2) #drop_last=False, pin_memory=False
        
        cur_res=test(cur_loader,model_restoration,CPU,out_bitdepth=8,baseline=BaseLine,PartitionBlock=PartitionBlock)
        print(name,' : \t ',cur_res)

print("over,test used ",(time.time()-test_start_time)/60," minutes!!!!!!!")

# torch.save({'epoch': epoch, 
#             'state_dict': model_restoration.state_dict(),
#             'optimizer' : optimizer.state_dict()
#             }, os.path.join(model_dir,"model_latest.pth"))         