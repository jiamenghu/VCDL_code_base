import argparse 
import copy


p=argparse.ArgumentParser()
p.add_argument('-q',help='QP,default:37',required =False)
p.add_argument('--lr',help='lr',required =False)
p.add_argument('--lr_min',help='lr_min',required =False)


p.add_argument('--bs',help='batch size',required =False)
p.add_argument('--ps',help='patch size',required =False)
p.add_argument('-L',help='load data before train ',required =False)
p.add_argument('--vps',help='val patch size',required =False)
p.add_argument('--tps',help='test patch size',required =False)
p.add_argument('--stride_frame',help='default 1',required =False)
p.add_argument('-r',help='Consecutive frames',required =False)


p.add_argument('--GPU',help='CPU /GPU',required =False)
p.add_argument('--local_rank', type=int, default=0, help='Distributed launcher requires.')  
p.add_argument('--CPU',help='CPU /GPU',action='store_true')


p.add_argument('--model',help='model path',required =False)
p.add_argument('--test',help='only test',action='store_true')
p.add_argument('--resume',help='retrain from last time checkpoint',action='store_true')
p.add_argument('-p',help='print_batch_info',action='store_true')
p.add_argument('--tranmodel',help='transformer model path',required =False)

p.add_argument('--trainset',help='DIV2K;BVIDVC;BVIDVC_A;',required =False)
p.add_argument('--test_frame',help='default 50:50; -1：all frames',required =False)
p.add_argument('--PB',help='PartitionBlock',action='store_true')



args = p.parse_args()

# ####参数集中营
# cfg={
#     'dataset':{

#     },
#     'network':{

#     },
#     'train':{

#     },
#     'val':{

#     },
#     'test':{

#     },
# }

############### mode ######################
QP=37
if args.q:
    QP=int(args.q)

GPU = []
if args.GPU:
    str_GPU=args.GPU.split(',')
    for i in str_GPU:
        if int(i)>=0 and int(i)<=7:
            GPU.append(int(i))

PartitionBlock=False
if args.PB:
    PartitionBlock=True
    

################### Train ################

model_path="./model/auto_saved_model.pth"
if args.model:
    model_path=args.model

PRINT_BATCH=False
if args.p:
    PRINT_BATCH=args.p

LR_INITIAL = 1e-4
if args.lr:
    LR_INITIAL=float(args.lr)

LR_DECAY_RATE=0.1 
LR_MIN = 1e-6
if args.lr_min:
    LR_MIN=float(args.lr_min)

# BETA1 = 0.5
PATIENCE_NUM = 5
THR_CLIP = 4   #梯度裁剪
MAX_EPOCHS = 10000
#EPOCH_DECAY = [100]

TRAIN_BATCH_SIZE = 64
if args.bs:
    TRAIN_BATCH_SIZE =int(args.bs)

VAL_BATCH_SIZE = 3
VAL_Interval=10


TEST_BATCH_SIZE = 1

TRAIN_PS = 64
if args.ps:
    TRAIN_PS =int(args.ps)
VAL_PS = -1
if args.vps:
    VAL_PS=int(args.vps)
TEST_PS= -1
if args.tps:
    TEST_PS =int(args.tps)


if VAL_PS==-1:
    VAL_BATCH_SIZE = 1
if TEST_PS==-1:
    TEST_BATCH_SIZE = 1  

DATA_AUG=True   #数据增强

RESUME = False  
if args.resume:
    RESUME =args.resume
#VAL_AFTER_EVERY = 3


STRIDE_FRAME=1
CONSEC_FRAME=3

if args.stride_frame:
    STRIDE_FRAME=int(args.stride_frame)
if args.r:
    CONSEC_FRAME=int(args.r)


################### Test  ################
BaseLine=False  ##是否有对比测试数据

CPU=False
if args.CPU:
    CPU=args.CPU

TEST=False
if args.test:
    TEST=args.test

ONLY_B=True   #不测试Class A

TEST_FRAMES=-1
if args.test_frame:
    TEST_FRAMES=int(args.test_frame)

############### Dataset #################

SERVER_ID=175  # 175,176                    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if SERVER_ID==176 :
    DIV2K_root_dir='/data1/Data/DIV2K/'
    BVIDVC_root_dir='/data1/Data/BVIDVC/'
    CTC_root_dir='/data1/Data/VTM_CTC/'
    Flicker2K_root_dir ='/data1/Data/Flicker2K/'
    
    div2k_root_dir='/data1/Data/DIV2K_HM/'
    ctc_hm_root_dir='/data1/Data/TEST_HM/'


    DATA_BVIDVC_LDP_F = { 
        'High' : BVIDVC_root_dir+'rawseq',
        'Reco' : BVIDVC_root_dir+'VTM7.0/LDP_'+str(QP)+'/vtms',
        'BitDepth' : {'High':10,'Reco':10},
        'SizePosition' : -4,
        'StrideFrame' : STRIDE_FRAME,
        'Frames':64
    }
    DATA_CTC8_LDP_F = { 
        'High' : CTC_root_dir+'rawseq_8',
        'Reco' : CTC_root_dir+'VTM7.0_8/LDP_'+str(QP)+'/vtms',
        'Vtms' : CTC_root_dir+'VTM7.0_8/LDP_'+str(QP)+'/vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }



    DATA_CTC10_LDP_F = { 
        'High' : CTC_root_dir+'rawseq_10',
        'Reco' : CTC_root_dir+'VTM7.0_10/AI_'+str(QP)+'/reco',
        'Vtms' : CTC_root_dir+'VTM7.0_10/AI_'+str(QP)+'/vtms',
        'BitDepth' : {'High':10,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }

    DATA_DIV2K_HM={
        'High' : div2k_root_dir+'div2k_high',
        'Reco' : div2k_root_dir+'AI_'+str(QP)+'_nf',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':1        
    }
    DATA_CTC_HM_AI = { 
        'High' : ctc_hm_root_dir+'original',
        'Reco' : ctc_hm_root_dir+'AI_'+str(QP)+'_nf',
        'Vtms' : ctc_hm_root_dir+'AI_'+str(QP)+'_hm',
        'BitDepth' : {'High':8,'Reco':8,'Vtms':8},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':-1
    }

    DATA_Flicker2K = { 
        'High' : Flicker2K_root_dir+'rawseq',
        'Reco' : Flicker2K_root_dir+'VTM7.0/AI_'+str(QP)+'/reco',
        'BitDepth' : {'High':8,'Reco':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames': 1
    }


    DATA_YUV108_LDP_NF={
        'High' : '/data1/Data/YUV108/YUV108_org',
        'Reco' : '/workspace/YUV108/HM169/LDP_'+str(QP)+'/reco_nf',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : STRIDE_FRAME,
        'Frames': 300
    }
    DATA_CTC_HM_LDP_NF = { 
        'High' : '/data1/Data/HM_CTC_50/original',
        'Reco' : '/data1/Data/HM_CTC_50/CTC_LDP/nf/LDP_'+str(QP)+'_nf',
        'Vtms' : '/data1/Data/HM_CTC_50/CTC_LDP/hm/LDP_'+str(QP)+'_nf',
        'BitDepth' : {'High':8,'Reco':8,'Vtms':8},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames': 50
    }
    DATA_YUV108_LDP_F={
        'High' : '/data1/Data/YUV108/YUV108_org',
        'Reco' : '/workspace/YUV108/HM169/LDP_'+str(QP)+'/hm_f',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : STRIDE_FRAME,
        'Frames': 300
    }
    DATA_CTC_HM_LDP_F = { 
        'High' : '/data1/Data/HM_CTC_50/original',
        'Reco' : '/data2/Data/jiamenghu_data/HM_CTC/hm/LDP_'+str(QP)+'_nf',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames': 50
    }

if SERVER_ID==175:
    DATA_YUV108_LDP_F={
        'High' : '/data/Data/YUV108/YUV108_org',
        'Reco' : '/data2/Data/jiamenghu_data/YUV108/HM169/LDP_'+str(QP)+'/hm_f',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : STRIDE_FRAME,
        'Frames': 300
    }
    DATA_CTC_HM_LDP_F = { 
        'High' : '/data2/Data/jiamenghu_data/HM_CTC/rawseq',
        'Reco' : '/data2/Data/jiamenghu_data/HM_CTC/HM169//LDP_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':8},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames': TEST_FRAMES
    }
    DATA_CTC8_VTM_LDP_F = { 
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_8',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_8/LDP_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }
    DATA_CTC10_VTM_LDP_F = {
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_10',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_10/LDP_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':10,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }
    DATA_YUV108_VTM_LDP_F = {
        'High' : '/data/Data/YUV108/VTM121/temp_org',#'/data/Data/YUV108/YUV108_org',
        'Reco' :'/data/Data/YUV108/VTM121/LDP_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }

    DATA_CTC8_VTM_RA_F = { 
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_8',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_8/RA_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }
    DATA_CTC10_VTM_RA_F = {
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_10',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_10/RA_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':10,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }
    DATA_YUV108_VTM_RA_F = {
        'High' : '/data/Data/YUV108/YUV108_org',#'/data/Data/YUV108/YUV108_org',
        'Reco' :'/data/Data/YUV108/VTM121/RA_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }

    DATA_CTC8_VTM_LDB_F = { 
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_8',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_8/LDB_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }

    DATA_CTC10_VTM_LDB_F = {
        'High' : '/data1/Data/VTM_CTC/'+'rawseq_10',
        'Reco' : '/data1/Data/VTM_CTC/VTM121_10/LDB_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':10,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }
    DATA_YUV108_VTM_LDB_F = {
        'High' : '/data/Data/YUV108/YUV108_org',#'/data/Data/YUV108/YUV108_org',
        'Reco' :'/data/Data/YUV108/VTM121/LDB_'+str(QP)+'/Vtms',
        'BitDepth' : {'High':8,'Reco':10,'Vtms':10},
        'SizePosition' : -2,
        'StrideFrame' : 1,
        'Frames':TEST_FRAMES
    }

EXTRA_CHANNEL=False
DATA_FLAG='YUV108_LDP_F'
#DATA_FLAG='DATA_YUV108_VTM_RA_F'

if args.trainset:
    DATA_FLAG=args.trainset

TRAIN_dir=[]
VAL_dir=[]
TEST_dir=[]

if SERVER_ID==176 and 'BVIDVC_LDP_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_BVIDVC_LDP_F)
if SERVER_ID==176 and 'YUV108_LDP_NF' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_LDP)
    VAL_dir.append(copy.deepcopy(DATA_CTC_HM_LDP))
    VAL_dir[0]['Frames']=10+CONSEC_FRAME
    TEST_dir=[DATA_CTC_HM_LDP]
if SERVER_ID==176 and 'YUV108_LDP_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_LDP_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC_HM_LDP_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC_HM_LDP_F]
    BaseLine=False

if SERVER_ID==175 and 'YUV108_LDP_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_LDP_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC_HM_LDP_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC_HM_LDP_F]
    BaseLine=False
if SERVER_ID==175 and 'DATA_YUV108_VTM_LDP_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_VTM_LDP_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC8_VTM_LDP_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC8_VTM_LDP_F,DATA_CTC10_VTM_LDP_F]
    BaseLine=False

if SERVER_ID==175 and 'YUV108_RA_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_RA_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC_HM_RA_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC_HM_RA_F]
    BaseLine=False
if SERVER_ID==175 and 'DATA_YUV108_VTM_RA_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_VTM_RA_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC8_VTM_RA_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC8_VTM_RA_F,DATA_CTC10_VTM_RA_F]
    BaseLine=False

if SERVER_ID==175 and 'YUV108_LDB_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_LDB_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC_HM_LDB_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC_HM_LDB_F]
    BaseLine=False
if SERVER_ID==175 and 'DATA_YUV108_VTM_LDB_F' in DATA_FLAG:
    TRAIN_dir.append(DATA_YUV108_VTM_LDB_F)
    VAL_dir.append(copy.deepcopy(DATA_CTC8_VTM_LDB_F))
    VAL_dir[0]['Frames']=5+CONSEC_FRAME
    TEST_dir=[DATA_CTC8_VTM_LDB_F,DATA_CTC10_VTM_LDB_F]
    BaseLine=False


if EXTRA_CHANNEL:
    DATA_CTC8['Extra_channel']=CTC_root_dir+'VTM7.0_8/AI_'+str(QP)+'/cumap'
    DATA_CTC8['BitDepth']['Extra_channel']=8

    DATA_CTC10['Extra_channel']=CTC_root_dir+'VTM7.0_10/AI_'+str(QP)+'/cumap'
    DATA_CTC10['BitDepth']['Extra_channel']=8


############# HM_CTC #####################
# VAL_dir.append(copy.deepcopy(DATA_CTC_HM_LDP))
# VAL_dir[0]['Frames']=10+CONSEC_FRAME
# TEST_dir=[DATA_CTC_HM_LDP]


############# VTM_CTC #####################
# VAL_dir.append(copy.deepcopy(DATA_CTC8))
# VAL_dir[0]['Frames']=3+CONSEC_FRAME
# TEST_dir=[DATA_CTC8,DATA_CTC10]


# VAL_dir.append(copy.deepcopy(DATA_CTC8_LDP_F))
# VAL_dir[0]['Frames']=4+CONSEC_FRAME
# TEST_dir=[DATA_CTC8_LDP_F,DATA_CTC10_LDP_F]
# BaseLine=False
# OUT_BitDepth=TEST_dir[0]['BitDepth']['Reco']


PRINT_SEETING=True
if PRINT_SEETING:
    print("########### SEETING ###############")
    print("QP :",QP,"; lr :",LR_INITIAL,"; min_lr :",LR_MIN,"; PATIENCE_NUM",PATIENCE_NUM,"THR_CLIP :",THR_CLIP)
    print('GPU:',GPU,' CPU:',CPU,' model_path:',model_path,' TEST_FRAMES: ',TEST_FRAMES)
    print("batch size :",TRAIN_BATCH_SIZE,'RESUME :',RESUME,'consec frames :',CONSEC_FRAME)
    print("train patch size :",TRAIN_PS,"; val patch size :",VAL_PS,"; test patch size :",TEST_PS)
    print("train batch size :",TRAIN_BATCH_SIZE,"; val batch size :",VAL_BATCH_SIZE,"; test batch size :",VAL_BATCH_SIZE)    
    print('StrideFrame: ',STRIDE_FRAME,' data_aug: ',DATA_AUG,' dataflag: ',DATA_FLAG)
    print("TRAIN_dir: " ,TRAIN_dir)
    print("VAL_dir: " ,VAL_dir)
    for t_d in TEST_dir:
        print("TEST_dir: " ,t_d)

if __name__ == '__main__' :
    pass