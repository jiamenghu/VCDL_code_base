import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/home/meng/code/multi_frame/DCN/")
from ops.dcn.deform_conv import ModulatedDeformConv,DeformConv

class ResiBlock_v1(nn.Module):
    def __init__(self,inchannel=64,outchannel=64,Act=None):
        super(ResiBlock_v1,self).__init__()
        self.inc=inchannel
        self.ouc=outchannel
        self.Act= Act if Act else nn.ReLU(inplace=True)

        self.base=nn.Sequential()
        self.base.add_module('bn_0',nn.BatchNorm2d(inchannel))
        self.base.add_module('relu_0',self.Act)
        self.base.add_module('conv_0',nn.Conv2d(inchannel,inchannel,3,padding=1))
        self.base.add_module('bn_1',nn.BatchNorm2d(inchannel))
        self.base.add_module('relu_1',self.Act)
        self.base.add_module('conv_1',nn.Conv2d(inchannel,outchannel,3,padding=1))
        
        self.connect=nn.Sequential()
        if inchannel !=outchannel:
            self.connect.add_module('c_conv_0',nn.Conv2d(inchannel,outchannel,1,padding=0))
            self.connect.add_module('c_relu_1',self.Act)

    def forward(self,x):
        y = self.base(x)
        if self.inc!=self.ouc:
            y = y+self.connect(x)
        else:
            y = y+x
        return y

class ConvLays(nn.Module):
    def __init__(self,inc=64,mic=64,ouc=64,lays=3,Act=None):
        super(ConvLays,self).__init__()

        self.inc=inc
        self.ouc=ouc
        self.mic= mic
        self.lays=lays
        self.Act= Act if Act else nn.ReLU(inplace=True)
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.c1=nn.Conv2d(self.inc,self.mic,3,1,1)
        self.ml=[]
        for i in range(0, self.lays-2):
            self.ml.append(nn.Conv2d(self.mic, self.mic, 3, stride=1, padding=3//2))
            self.ml.append(self.Act)
        self.cn=nn.Sequential(*self.ml)
        self.c2=nn.Conv2d(self.mic,self.ouc,3,1,1)        


    def forward(self,fea):

        f1=self.Act(self.c1(fea))
        f2=self.cn(f1)
        f3=self.Act(self.c2(f2))
        return f3

class ResiConvLays(nn.Module):
    def __init__(self,inc=64,mic=64,ouc=64,lays=3,Act=None):
        super(ConvLays,self).__init__()

        self.inc=inc
        self.ouc=ouc
        self.mic= mic
        self.lays=lays
        self.Act= Act if Act else nn.ReLU(inplace=True)
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.c1=ResiBlock_v1(self.inc,self.mic,self.Act)
        self.ml=[]
        for i in range(0, self.lays-2):
            self.ml.append(ResiBlock_v1(self.mic, self.mic, self.Act))
            self.ml.append(self.Act)
        self.cn=nn.Sequential(*self.ml)
        self.c2=ResiBlock_v1(self.mic,self.ouc,self.Act)

    def forward(self,fea):

        f1=self.c1(fea)
        f2=self.cn(f1)
        f3=self.c2(f2)
        return f3

class FEN(nn.Module):
    def __init__(self,inc=1,ouc=[64,64,64],Act=None):
        super(FEN,self).__init__()
        print('feaNN ',self.__class__.__name__)

        if Act:
            self.Act=Act
        else:   
            self.Act=nn.ReLU()
        self.ouc=ouc
        self.inc=inc
        self.c11=nn.Conv2d(self.inc,   self.ouc[0],3,1,1)
        self.c12=nn.Conv2d(self.ouc[0],self.ouc[0],3,1,1)
        self.c13=nn.Conv2d(self.ouc[0],self.ouc[0],3,1,1)

        self.c21=nn.Conv2d(self.ouc[0],self.ouc[1],3,2,1)
        self.c22=nn.Conv2d(self.ouc[1],self.ouc[1],3,1,1)
        self.c23=nn.Conv2d(self.ouc[1],self.ouc[1],3,1,1)

        self.c31=nn.Conv2d(self.ouc[1],self.ouc[2],3,2,1)
        self.c32=nn.Conv2d(self.ouc[2],self.ouc[2],3,1,1)
        self.c33=nn.Conv2d(self.ouc[2],self.ouc[2],3,1,1)

    def forward(self,img):

        f11=self.Act(self.c11(img))
        f12=self.Act(self.c12(f11))
        f13=self.Act(self.c13(f12))

        f21=self.Act(self.c21(f13))
        f22=self.Act(self.c22(f21))
        f23=self.Act(self.c23(f22))

        f31=self.Act(self.c31(f23))
        f32=self.Act(self.c32(f31))
        f33=self.Act(self.c33(f32))

        return f13,f23,f33      


class OffsetNet(nn.Module):
    def __init__(self,inc=3*64,mic=64,ouc=4*1*3*9,upsample=None,lays=3,Act=None):
        super(OffsetNet,self).__init__()

        self.inc=inc
        self.ouc=ouc
        self.mic= mic
        self.lays=lays
        self.upsample=upsample
        self.Act= Act if Act else nn.ReLU(inplace=True)
        #self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.c1=nn.Conv2d(self.inc,self.mic,3,1,1)
        self.ml=[]
        for i in range(0, self.lays-2):
            self.ml.append(nn.Conv2d(self.mic, self.mic, 3, stride=1, padding=3//2))
            self.ml.append(self.Act)
        self.cn=nn.Sequential(*self.ml)

        self.c2=nn.Conv2d(self.mic,self.ouc,3,1,1)

    def forward(self,fea):

        f1=self.Act(self.c1(fea))
        f2=self.cn(f1)
        f3=self.c2(f2)
        return f3

class Warp_deform(nn.Module):
    def __init__(self,inc=64,ouc=64,groups=1):
        super(Warp_deform,self).__init__()
        self.warp=DeformConv(inc,ouc,3, padding=1, deformable_groups=groups)
    
    def forward(self,fea,offset):
        return self.warp(fea.contiguous(),offset.contiguous())

class BasicOPN(nn.Module):
    def __init__(self,curf_c=64,reff_c=2*64,init_groups=1,groups=1,lays=3,Act=None):
        super(BasicOPN,self).__init__()

        self.init_groups=init_groups
 
        self.warp=Warp_deform(inc=reff_c//2,ouc=reff_c//2,groups=init_groups)
        self.opn=OffsetNet(inc=curf_c+reff_c,mic=curf_c,ouc=2*groups*2*9,lays=lays,Act=Act)

    def forward(self,curf,reff,initoff=None):
        B,C,H,W=reff.shape

        if initoff==None:
            offset=self.opn(torch.cat([curf,reff],1))
            return offset
        initoff=initoff.contiguous().reshape(B*2,-1,H,W)

        reff_warp=self.warp(reff.contiguous().reshape(B*2,-1,H,W),initoff).contiguous().reshape(B,-1,H,W)
        offset=self.opn(torch.cat([curf,reff_warp],1))

        initoff=initoff[:,0:1*2*9,...]

        initoff=initoff.contiguous().reshape(B,-1,H,W)
        offset=self.ADDflow(offset,initoff)
        return offset

    def ADDflow(self,res,flow):
        B,C,H,W=res.shape
        _,Cf,_,_=flow.shape
        for i in range(C//Cf):
            res[:,i*Cf:(i+1)*Cf,...]+=flow
        return res

class STO(nn.Module):
    def __init__(self,ops):
        super(STO,self).__init__()
        self.T=ops['frames']
        self.groups=ops['pppn']['groups']
        self.lays=ops['pppn']['offset']['lays']
        self.Act=ops['Act']
        self.sfc=ops['fxn']['ouc']

        self.s0_opns=nn.ModuleList()
        self.s1_opns=nn.ModuleList()
        self.s2_opns=nn.ModuleList()
        
        for i in range(self.T//2):
            self.s0_opns.append(BasicOPN(self.sfc[0],2*self.sfc[0],1,1,3))
            self.s1_opns.append(BasicOPN(self.sfc[1],2*self.sfc[1],1,1,3))
            self.s2_opns.append(BasicOPN(self.sfc[2],2*self.sfc[2],1,1,3))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self,fs0,fs1,fs2):
        '''
        fs0 ：[t0,t1,t2,t3]
        fs1 ：[t0,t1,t2,t3]
        fs2 : [t0,t1,t2,t3]
        '''

        #2
        offset_2=[]
        for i in range(1,self.T//2+1):
            if i==1:
                offset_2.append(self.s2_opns[i-1](fs2[0],fs2[1]))
            else:
                offset_2.append(self.s2_opns[i-1](fs2[0],fs2[i], (i/(i-1))*offset_2[i-2]))

        for i in range(len(offset_2)):
            offset_2[i]=2*self.upsample(offset_2[i])


        #1
        offset_1=[]
        for i in range(1,self.T//2+1):
            if i==1:
                offset_1.append(self.s1_opns[i-1](fs1[0],fs1[1],offset_2[0]))
            else:
                offset_1.append(self.s1_opns[i-1](fs1[0],fs1[i],(offset_2[i-1]+(i/(i-1))*offset_1[i-2])/2))

        for i in range(len(offset_1)):
            offset_1[i]=2*self.upsample(offset_1[i])        

        #0

        offset_0=[]
        for i in range(1,self.T//2+1):
            if i==1:
                offset_0.append(self.s0_opns[i-1](fs0[0],fs0[1],offset_1[0]))
            else:
                offset_0.append(self.s0_opns[i-1](fs0[0],fs0[i],(offset_1[i-1]+(i/(i-1))*offset_0[i-2])/2))
        
        #offset=torch.cat(offset_0,1)

        return offset_0  

class OTA(nn.Module):
    def __init__(self,T,fea_inc,off_inc,ouc):
        super(OTA,self).__init__()

        self.T=T
        
        self.n1=nn.ModuleList()
        for i in range(T//2):
            self.n1.append(ConvLays(inc=off_inc+fea_inc,mic=fea_inc,ouc=fea_inc,lays=3))
        self.n2=ConvLays(inc=fea_inc*(T//2),mic=64,ouc=fea_inc*(T//2),lays=3)
        
    
    def forward(self,fea,off):

        att_fea=[]
        for i in range(self.T//2):
            att_fea.append(self.n1[i](torch.cat([fea[i],off[i]],1)))
        
        att_fea=self.n2(torch.cat(att_fea,1))
        #print('  ',torch.cat(fea,1).shape,att_fea.shape)
        return torch.cat(fea,1)*F.softmax(att_fea,1)

class TAN(nn.Module):
    def __init__(self,T,inc,ouc,groups):
        super(TAN,self).__init__()
        print('NN: ',self.__class__.__name__)

        self.T=T
        self.inc=inc
        self.ouc=ouc
        self.groups=groups

        self.dcns=nn.ModuleList()

        for i in range(T//2):
            self.dcns.append(DeformConv(2*self.inc,2*self.ouc,3, padding=1, deformable_groups=2))
        
        self.att=OTA(T=T,fea_inc=2*ouc, off_inc=18*2,ouc=2*ouc)    

    def forward(self,fea,off):
        
        wfea=[]
        for i in range(self.T//2):
            #print(fea[i].shape,off[i].shape)
            wfea.append(self.dcns[i](fea[i].contiguous(),off[i].contiguous()))

        #att_wfea=self.att(wfea,off)
        att_wfea= torch.cat(wfea,1)
        return att_wfea

class MSTAN(nn.Module):
    def __init__(self,T,inc,ouc,groups):
        super(MSTAN,self).__init__()   
        self.T=T
        self.inc=inc
        self.ouc=ouc
        self.groups=groups

        self.dcn0=TAN(T=T,inc=inc,ouc=ouc,groups=groups)
        self.dcn1=TAN(T=T,inc=inc,ouc=ouc,groups=groups)
        self.dcn2=TAN(T=T,inc=inc,ouc=ouc,groups=groups)

    def forward(self,fea,off):
        
        fea1=[]
        fea2=[]
        off1=[]
        off2=[]

        for i in range(len(fea)):
            fea1.append(F.avg_pool2d(fea[i],(2,2)))
            fea2.append(F.avg_pool2d(fea1[i],(2,2)))

        for i in range(len(off)):
            off1.append(F.avg_pool2d(off[i],(2,2))/2)
            off2.append(F.avg_pool2d(off1[i],(2,2))/2)

        wf0=self.dcn0(fea[1:],off)
        wf1=self.dcn1(fea1[1:],off1)
        wf2=self.dcn2(fea2[1:],off2)

        return torch.cat([wf0,fea[0]],1),torch.cat([wf1,fea1[0]],1),torch.cat([wf2,fea2[0]],1)

class MSIRN(nn.Module):
    def __init__(self,ops):
        super(MSIRN,self).__init__()
        print('QE NN: ',self.__class__.__name__)

        self.T=ops['frames']
        self.inc=ops['qen']['inc']
        self.ouc=ops['qen']['ouc']
        self.mic=ops['qen']['mic']
        self.lays=ops['qen']['lays']

        self.Act=ops['Act']

        self.qe_subnet0=ConvLays(inc=self.inc,mic=self.mic,ouc=self.ouc,lays=self.lays,Act=self.Act)
        self.qe_subnet1=ConvLays(inc=self.inc,mic=self.mic,ouc=self.ouc,lays=self.lays,Act=self.Act)
        self.qe_subnet2=ConvLays(inc=self.inc,mic=self.mic,ouc=self.ouc,lays=self.lays,Act=self.Act)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self,wf0,wf1,wf2):


        out0=self.qe_subnet0(wf0)
        out1=self.qe_subnet1(wf1)
        out2=self.qe_subnet2(wf2)
        out=out0+self.upsample(out1+self.upsample(out2))

        return out

 
class STO_MGO(nn.Module):
    def __init__(self,T):
        super(STO_MGO,self).__init__()
        print('NN: ',self.__class__.__name__)
    
        self.Act=nn.LeakyReLU(negative_slope=0.1)
        ops={
            'frames':T,'Act':self.Act,
            'fxn':{
                'inc':1,'ouc':[64,64,64],},
            'pppn':{
                'offset':{'inc':64,'lays':4,},
                'groups':[1,1,1],
            },
            'tan':{
                'inc':64,'ouc':64,'groups':T-1
            },
            'qen':{
                'inc':T*64,'ouc':1,'mic':64,'lays':8,
            }
        }

        self.fen=FEN(inc=ops['fxn']['inc'],ouc=ops['fxn']['ouc'],Act=self.Act)
        self.stopn=STO(ops)
        self.mstan=MSTAN(T=T,inc=ops['tan']['inc'],ouc=ops['tan']['ouc'],groups=ops['tan']['groups'])
        self.msirn=MSIRN(ops)
        
        for model_restoration in [self.fen,self.stopn,self.mstan,self.msirn]:
            print(model_restoration.__class__.__name__,sum(p.numel() for p in model_restoration.parameters()))

    def forward(self,imgs):
        T=len(imgs)
        B,_,H,W=imgs[0].shape

        img=torch.cat(imgs,1).reshape(B*T,1,H,W)
        #print(B,H,W,img.shape)
        fd0,fd1,fd2=self.fen(img)

        fd0=torch.chunk(fd0.reshape(B,-1,H,W),T,1)
        fd1=torch.chunk(fd1.reshape(B,-1,H//2,W//2),T,1)
        fd2=torch.chunk(fd2.reshape(B,-1,H//4,W//4),T,1)

        ##
        fs0=[]
        fs1=[]
        fs2=[]
        #fs=:[t0, [t-1,t+1],[t-2,t+2],...]
        for i in range(T//2+1):
            if i==0:
                fs0.append(fd0[T//2])
                fs1.append(fd1[T//2])
                fs2.append(fd2[T//2])
            else:
                fs0.append(torch.cat([fd0[T//2-i],fd0[T//2+i]],1))
                fs1.append(torch.cat([fd1[T//2-i],fd1[T//2+i]],1))
                fs2.append(torch.cat([fd2[T//2-i],fd2[T//2+i]],1))    


        off=self.stopn(fs0,fs1,fs2)

        wf0,wf1,wf2=self.mstan(fs0,off)
        
        enf=self.msirn(wf0,wf1,wf2)

        return enf+imgs[T//2]   
