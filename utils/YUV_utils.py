'''
author: Menghu Jia
date  : 2021/1/1
code for YUV file operations
'''

import numpy as np
import math, os
from functools import partial



def load_file_list(directory,filter=''):
    '''
    读取文件名称及其路径
    filter:过滤器
    '''
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        if filter and filter not in filename:
            continue 
        list.append(os.path.join(directory,filename))
    return sorted(list)

def get_seq_dir_list(DATA_dir_list,extra_channel,TRAIN=True,baseline=False) : #element seqencce dir 
    '''
    返回包含所有序列的列表
    每个元素包含一个序列内容：
    {Reco,High,Vtms*,Extra_channel*,BitDepth,SizePosition,StrideFrame,Frames}
    '''
    if not isinstance(DATA_dir_list, list):
        DATA_dir_list = [DATA_dir_list]
    seq_dir_list=[]
    for DATA_dir in DATA_dir_list: 
        reco_seq_dir=load_file_list(DATA_dir['Reco'])
        high_seq_dir=load_file_list(DATA_dir['High'])
        assert len(reco_seq_dir) == len(high_seq_dir), "数据有误，reco_seq("+str(len(reco_seq_dir))+") 和 high_seq("+str(len(high_seq_dir))+") 数量不一致！！"
        if extra_channel:
            extra_seq_dir=load_file_list(DATA_dir['Extra_channel'])
            assert len(reco_seq_dir) == len(extra_seq_dir), "数据有误，reco_seq("+str(len(reco_seq_dir))+") 和 extra_seq("+str(len(extra_seq_dir))+") 数量不一致！！"

        if not TRAIN and baseline:
            assert 'Vtms' in DATA_dir,"TESTING, 没有Vtms"
            vtms_seq_file=load_file_list(DATA_dir['Vtms'])

        #element : {Reco,High,Vtms*,Extra_channel*,BitDepth,SizePosition,StrideFrame,Frames}
        for i in range(len(reco_seq_dir)):
            ############筛选HM的测试集
            if 'HM_CTC' in reco_seq_dir[i]:
                if 'BasketballDrillText' in reco_seq_dir[i] or 'ChinaSpeed' in reco_seq_dir[i] or  'SlideEditing' in reco_seq_dir[i] or 'SlideShow' in reco_seq_dir[i] or  'vidyo' in reco_seq_dir[i] :
                    continue
            ##########

            seq_dir_list.append({'Reco':reco_seq_dir[i],'High':high_seq_dir[i]})
            if extra_channel: seq_dir_list[-1]['Extra_channel']=extra_seq_dir[i]
            if not TRAIN and baseline:  seq_dir_list[-1]['Vtms']=vtms_seq_file[i]
            seq_dir_list[-1]['BitDepth']=DATA_dir['BitDepth']
            seq_dir_list[-1]['SizePosition']=DATA_dir['SizePosition']
            seq_dir_list[-1]['StrideFrame']=DATA_dir['StrideFrame']
            seq_dir_list[-1]['Frames']=DATA_dir['Frames']    
        
    return seq_dir_list

def get_elem_dir_list(seq_dir_list, elem_params):  #element data_patch dir
    '''
    返回包含所有网络输入基本元素列表
    params:elem_params.consec_frames
    每个元素包含一个patch对：
    {Reco,High,Vtms*,Extra_channel*,BitDepth,SizePosition,StrideFrame,Frames} 
    '''

    elem_dir_list = []
    for seq_dir in seq_dir_list:
        
        ###HM filter#####

        ################
        width,height=getWH(seq_dir['High'],seq_dir['SizePosition'])

        reco_frame_num=count_num_frames(seq_dir['Reco'],width,height,seq_dir['BitDepth']['Reco'])

        if seq_dir['Frames']==-1 or seq_dir['Frames']>reco_frame_num:   seq_dir['Frames']=reco_frame_num

        for i in range(0,seq_dir['Frames']-elem_params['consec_frames'],seq_dir['StrideFrame']):

                elem_dir_list.append({"Reco":seq_dir['Reco'],"High":seq_dir['High'],'consec_fra':elem_params['consec_frames'],
                                    "lq_frame_index":i,"gt_frame_index":int((i+i+elem_params['consec_frames'])//2),
                                    'width':width,'height':height,'BitDepth':seq_dir["BitDepth"]})
                if not elem_params['train'] and elem_params['baseline']: 
                    elem_dir_list[-1]['Vtms']=seq_dir['Vtms']
                if elem_params['extra_channel'] : elem_dir_list[-1]['Extra_channel']=seq_dir['Extra_channel']

    return elem_dir_list

def getWH(SeqPath,SizePosition):
    '''
    读取yuv宽高(命名规则：XXXX_416x240_XX.yuv)
    '''
    wxh = os.path.basename(SeqPath).split('_')[SizePosition]
    w, h = wxh.split('x')
    return int(w), int(h)

#获得一帧patch最大数量（stride=PATCH_size）
def count_num_patchs(width,height,PATCH_SIZE,stride_w=0,stride_h=0):
    w_n=width//(PATCH_SIZE+stride_w)
    h_n=height//(PATCH_SIZE+stride_h)
    return h_n*w_n

#获取PATCH位置
def get_patch_location(height,width,index,PATCH_SIZE):
    #print(width,height)
    w_n=width//PATCH_SIZE
    h_n=height//PATCH_SIZE
    i=index//w_n
    j=index-i*w_n
    return i*PATCH_SIZE,j*PATCH_SIZE


def count_num_frames(filename,width,height,BitDepth):
    '''
    返回输入的yuv文件(只支持420格式)的帧数(8 bit or 10 bit)
    :param filename : YUV file path.  
    :param width : frame width.  
    :param height : frame height.  
    :param BitDepth : YUV bitdepth(8bit/10bit).  
    :return : frame numers.  
    '''
    fp = open(filename, 'rb')
    b=1 if BitDepth==8 else 2               
    framesize = b*height * width * 3 // 2   # 一帧图像所含的二进制位数 
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部
    ps = fp.tell()  # 当前文件指针位置
    numfrm = ps // framesize  # 计算输出帧数
    return numfrm



def read_frame_yuv(seq_path, w,h, BitDepth,startfrm,tot_frm=1,only_y=True,yuv_format='420p'):

    if yuv_format == '420p':
        uv_h, uv_w = h // 2, w // 2
    elif yuv_format == '444p':
        uv_h, uv_w = h, w
    else:
        raise Exception('yuv_format not supported.')

    if BitDepth == 8:
        pixel_bit=1
    elif BitDepth == 10:
        pixel_bit =2
    else:
        raise Exception('BitDepth not supported.')

    y_size, u_size, v_size = h * w, uv_h * uv_w, uv_h * uv_w
    frame_size = (y_size + u_size + v_size) * pixel_bit  # 一帧图像大小 / 字节
    
    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=(np.uint8 if BitDepth==8 else np.int16))
    if not only_y:
        u_seq = np.zeros((tot_frm, uv_h, uv_w), dtype=(np.uint8 if BitDepth==8 else np.int16))
        v_seq = np.zeros((tot_frm, uv_h, uv_w), dtype=(np.uint8 if BitDepth==8 else np.int16))

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(frame_size * (startfrm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), \
                    count=u_size).reshape(uv_h, uv_w)
                v_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), \
                    count=v_size).reshape(uv_h, uv_w)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq
def read_patch_yuv(seq_path, w,h, BitDepth,startfrm,patch_pos,patch_size,tot_frm=1,only_y=True,yuv_format='420p'):
    #print(seq_path, w,h, BitDepth,startfrm,patch_pos,patch_size,tot_frm)
    px,py=patch_pos[0],patch_pos[1]  #patch 左上角坐标 px水平位置，py垂直位置
    if yuv_format == '420p':
        uv_px,uv_py=px//2,py//2
        uv_h, uv_w = h // 2, w // 2
        uv_p=patch_size//2
    elif yuv_format == '444p':
        uv_px,uv_py=px,py
        uv_h, uv_w = h, w
        uv_p=patch_size
    else:
        raise Exception('yuv_format not supported.')

    if BitDepth == 8:
        pixel_bit=1
    elif BitDepth == 10:
        pixel_bit =2
    else:
        raise Exception('BitDepth not supported.')

    y_size, u_size, v_size = h * w, uv_h * uv_w, uv_h * uv_w
    frame_size = (y_size + u_size + v_size)   # 一帧图像像素个数
    
    # init
    y_seq = np.zeros((tot_frm, patch_size, patch_size), dtype=(np.uint8 if BitDepth==8 else np.int16))
    if not only_y:
        u_seq = np.zeros((tot_frm, uv_p, uv_p), dtype=(np.uint8 if BitDepth==8 else np.int16))
        v_seq = np.zeros((tot_frm, uv_p, uv_p), dtype=(np.uint8 if BitDepth==8 else np.int16))

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int((frame_size * (startfrm + i)+py*w)* pixel_bit), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), count=patch_size*w).reshape(patch_size, w)
            if only_y:
                y_seq[i, ...] = y_frm[...,px:px+patch_size]
            else:
                raise Exception('patch UV not supported.')
            #     u_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), \
            #         count=u_size).reshape(uv_h, uv_w)
            #     v_frm = np.fromfile(fp, dtype=(np.uint8 if BitDepth==8 else np.int16), \
            #         count=v_size).reshape(uv_h, uv_w)
            #     y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq


def write_ycbcr(y, cb, cr, vid_path):
    with open(vid_path, 'ab') as fp:
        for ite_frm in range(len(y)):
            fp.write(y[ite_frm].reshape(((y[0].shape[0])*(y[0].shape[1]), )))
            fp.write(cb[ite_frm].reshape(((cb[0].shape[0])*(cb[0].shape[1]), )))
            fp.write(cr[ite_frm].reshape(((cr[0].shape[0])*(cr[0].shape[1]), )))


if __name__=="__main__":
    '''
    {'Reco': '/data/Data/VTM_CTC/VTM7.0_8/AI_37/reco/BQMall_832x480_60_AI_37.yuv', 
    'High': '/data/Data/VTM_CTC/rawseq_8/BQMall_832x480_60.yuv', 
    'frame_index': 2, 'width': 832, 'height': 480, 
    'BitDepth': {'High': 8, 'Reco': 10, 'Vtms': 10}, 
    'patch_size': -1}
    '''
    pass





'''

old version
'''
# def get_elem_dir_list(seq_dir_list, elem_params):  #element data_patch dir
#     '''
#     返回包含所有网络输入基本元素列表
#     params:elem_params.patch_size,elem_params.consec_frames
#     每个元素包含一个patch对：
#     {Reco,High,Vtms*,Extra_channel*,BitDepth,SizePosition,StrideFrame,Frames} 
#     '''
#     elem_dir_list = []
#     for seq_dir in seq_dir_list:
#         width,height=getWH(seq_dir['High'],seq_dir['SizePosition'])
#         reco_frame_num=count_num_frames(seq_dir['Reco'],width,height,seq_dir['BitDepth']['Reco'])
#         if seq_dir['Frames']>reco_frame_num:   seq_dir['Frames']=reco_frame_num
#         patch_num_onefra=count_num_patchs(width,height,elem_params['patch_size'])

#         for i in range(0,seq_dir['Frames']-elem_params['consec_frames'],seq_dir['StrideFrame']):
#             if elem_params['patch_size']==-1:
#                 elem_dir_list.append({"Reco":seq_dir['Reco'],"High":seq_dir['High'],"frame_index":i,
#                                         'width':width,'height':height,'BitDepth':seq_dir["BitDepth"],
#                                         'patch_size':elem_params['patch_size']})
#                 if 'Vtms' in seq_dir: elem_dir_list[-1]['Vtms']=seq_dir['Vtms']
#                 if 'Extra_channel' in seq_dir: elem_dir_list[-1]['Extra_channel']=seq_dir['Extra_channel']

#             else:
#                 for j in range(0,patch_num_onefra,1):
#                     patch_px,patch_py=get_patch_location(height,width,j,elem_params['patch_size'])
#                     elem_dir_list.append({"Reco":seq_dir['Reco'],"High":seq_dir['High'],"frame_index":i,
#                                         'width':width,'height':height,'BitDepth':seq_dir["BitDepth"],
#                                         "patch_pos":[patch_px,patch_py],'patch_size':elem_params['patch_size']})                    
#                     if 'Vtms' in seq_dir: elem_dir_list[-1]['Vtms']=seq_dir['Vtms']
#                     if 'Extra_channel' in seq_dir: elem_dir_list[-1]['Extra_channel']=seq_dir['Extra_channel']

#     return elem_dir_list

