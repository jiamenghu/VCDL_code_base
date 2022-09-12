import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import time
from datetime import datetime
def pre_train(train_loader,model,pargms_name):
    '''
    预训练，自适应修改最后一层权值大小，使输出std分布与真实std一致
    '''

    print('pre trian starting')
    with torch.no_grad():
        model.eval()
        factor=0
        i=0      
        for step_i,batch in enumerate(train_loader):
            i+=1
            reco_Y,high_Y=batch['Reco_Y'],batch['High_Y']

            reco_Ys=torch.split(reco_Y,1,dim=1)

            #extra_Y=batch['extra_Y]
            #resi_Y=Variable(resi_Y.cuda(), requires_grad=False)
            reco_Ys=[Variable(reco_Y.cuda(), requires_grad=False) for reco_Y in reco_Ys] 
            high_Y=Variable(high_Y.cuda(), requires_grad=False)
            
            output = model(reco_Ys)

            output_resi=output-reco_Ys[len(reco_Ys)//2]
            highresi_std=torch.std(high_Y-reco_Ys[len(reco_Ys)//2])
            outresi_std=torch.std(output_resi)
            factor += highresi_std/outresi_std
            #print(i)
            #print(3,i)
            if i==15:
                for name,param in model.named_parameters():
                    if pargms_name == name:  #########
                    #if 'base.base_conv_15.weight' == name:
                        print(step_i,factor/i,'old:',torch.std(param.data),end=' ')
                        param.data *= factor/i
                        print('new:',torch.std(param.data),' highresi_std:',highresi_std,' outresi_std: ',outresi_std,flush=True)
                if factor/i<=1.1 and factor/i>=0.9:
                    break
                factor=0
                i=0
            if step_i>100:
                break
        print('pre_train done')

def train(epoch,train_loader,model,criterion,optimizer,train_opts):
    model.train()
    psnr_out=0
    psnr_in=0
    loss_avg=0
    batch_num=0

    for step_i,batch in enumerate(train_loader):              #tqdm(train_loader), 0):

        reco_Y,high_Y=batch['Reco_Y'],batch['High_Y']
        #extra_Y=batch['extra_Y]
        reco_Ys=torch.split(reco_Y,1,dim=1)

        batch_psnr_in=torchPSNR(reco_Ys[len(reco_Ys)//2],high_Y).item()

        # resi_Y=Variable(resi_Y.cuda(), requires_grad=True)
        reco_Ys=[Variable(reco_Y.cuda(), requires_grad=True) for reco_Y in reco_Ys] 
        high_Y=Variable(high_Y.cuda(), requires_grad=False)

        model.zero_grad()
        output= model(reco_Ys)

        loss = criterion(output,high_Y)

        loss_avg+=loss.item()

        batch_psnr_out=torchPSNR(output,high_Y).item()
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),train_opts['thr_clip'])

        optimizer.step()

        if train_opts['print_batch'] :
            print(step_i,'\t',loss.item(),'\t',batch_psnr_out,'\t',batch_psnr_in,'\t',batch_psnr_out-batch_psnr_in)
        psnr_in+=batch_psnr_in
        psnr_out+=batch_psnr_out
        batch_num+=1
        del reco_Ys
        torch.cuda.empty_cache()

    psnr_out_avg=psnr_out/batch_num
    psnr_in_avg = psnr_in/batch_num
    gain=psnr_out_avg-psnr_in_avg
    loss_avg /= batch_num
    print("{}\t train epoch:{}\tloss:{}\tgain:{}\tout:{}\tin:{}\t".format(datetime.now(),epoch,loss_avg,gain,psnr_out_avg,psnr_in_avg))

def valid(epoch,valid_loader,model,criterion,baseline,PartitionBlock=False):
    with torch.no_grad():
        model.eval()
        psnr_out=0
        psnr_in=0
        psnr_bench=0
        loss_avg=0
        batch_num=0
        #psnr_vtm=[]
        for step_i,batch in enumerate(valid_loader):
            reco_Y,high_Y=batch['Reco_Y'],batch['High_Y']
            reco_Ys=torch.split(reco_Y,1,dim=1)

            batch_psnr_in=torchPSNR(reco_Ys[len(reco_Ys)//2],high_Y).numpy()

            reco_Ys=[Variable(reco_Y.cuda(), requires_grad=False) for reco_Y in reco_Ys] 
            high_Y=Variable(high_Y.cuda(), requires_grad=False)

            PartitionBlock=True
            if PartitionBlock and reco_Y.shape[3]>=1280:
                PMB=PartitionMergeBlock(reco_Ys)
                out_blocks=[]
                for blocks in PMB:
                    if blocks[0] .shape[3]>=1088:
                        sub_PMB=PartitionMergeBlock(blocks)
                        sub_out_blocks=[]
                        for sub_blocks in sub_PMB:
                            sub_out_blocks.append(model(sub_blocks))
                        sub_output=sub_PMB.merge(sub_out_blocks) 
                        out_blocks.append(sub_output) 
                    else :
                        out_blocks.append(model(blocks))

                output=PMB.merge(out_blocks)
            else:
                output= model(reco_Ys)

            loss = criterion(output,high_Y)
            #print(1)
            loss_avg+=loss.item()

            batch_psnr_out=torchPSNR(output,high_Y).item()   
            #torch.cuda.empty_cache()
            batch_num+=1
            psnr_in+=batch_psnr_in
            psnr_out+=batch_psnr_out

        psnr_out_avg=psnr_out/(batch_num)
        psnr_in_avg = psnr_in/(batch_num)
        gain_out=psnr_out_avg-psnr_in_avg
        loss_avg /= batch_num
    return {"loss":loss_avg,"gain":gain_out,"out":psnr_out_avg,"in":psnr_in_avg}

def test(test_loader,model,CPU=False,out_bitdepth=10,baseline=True,PartitionBlock=False):
    with torch.no_grad():
        model.eval()
        psnr_out=0
        psnr_in=0
        psnr_bench=0
        batch_num=0
        if baseline: psnr_bench=0

        for step_i,batch in enumerate(test_loader):
            reco_Y,high_Y=batch['Reco_Y'],batch['High_Y']
            if baseline: vtms_Y=batch['Vtms_Y']

            reco_Ys=torch.split(reco_Y,1,dim=1)

            batch_psnr_in=numpyPSNR(denormalize_zero(reco_Ys[len(reco_Ys)//2].numpy(),BitDepth=out_bitdepth),\
                                        high_Y.numpy(),max_value=2**out_bitdepth-1)
            if baseline:
                batch_psnr_bench=numpyPSNR(vtms_Y.numpy(),high_Y.numpy(),max_value=2**out_bitdepth-1)

            if CPU:
                reco_Ys=[Variable(reco_Y, requires_grad=False) for reco_Y in reco_Ys] 
                reco_Ys=[reco_Y.type(torch.FloatTensor) for reco_Y in reco_Ys] 
            else:
                reco_Ys=[Variable(reco_Y.cuda(), requires_grad=False) for reco_Y in reco_Ys] 
                reco_Ys = [reco_Y.type(torch.cuda.FloatTensor) for reco_Y in reco_Ys] 
            

            if PartitionBlock and reco_Y.shape[3]>=1280:
                PMB=PartitionMergeBlock(reco_Ys)
                out_blocks=[]
                for blocks in PMB:
                    if blocks[0] .shape[3]>=1088:
                        sub_PMB=PartitionMergeBlock(blocks)
                        sub_out_blocks=[]
                        for sub_blocks in sub_PMB:
                            sub_out_blocks.append(model(sub_blocks))
                        sub_output=sub_PMB.merge(sub_out_blocks) 
                        out_blocks.append(sub_output) 
                    else :
                        out_blocks.append(model(blocks))
                output=PMB.merge(out_blocks)
            else:
                output= model(reco_Ys)

            if CPU:       
                batch_psnr_out=numpyPSNR(denormalize_zero(output.numpy(),BitDepth=out_bitdepth), \
                                        high_Y.numpy(),max_value=2**out_bitdepth-1)
            else:
                batch_psnr_out=numpyPSNR(denormalize_zero(output.cpu().detach().numpy(),BitDepth=out_bitdepth),
                                        high_Y.numpy(),max_value=2**out_bitdepth-1)
                torch.cuda.empty_cache()
            batch_num+=1
            psnr_in+=batch_psnr_in
            psnr_out+=batch_psnr_out
            if baseline:
                psnr_bench+=batch_psnr_bench
            del reco_Y
            del high_Y
            del reco_Ys

        psnr_out_avg=psnr_out/batch_num
        psnr_in_avg = psnr_in/batch_num
        if baseline:
            psnr_bench_avg=psnr_bench/batch_num
            gain_bench=psnr_bench_avg-psnr_in_avg
        else:
            gain_bench=0
            psnr_bench_avg=0
        gain_out=psnr_out_avg-psnr_in_avg
        gain = gain_out-gain_bench
    return {"out":psnr_out_avg,"in":psnr_in_avg,"bench":psnr_bench_avg,"gain_bench":gain_bench,"gain_out":gain_out,"gain":gain}


# for i, data in enumerate(tqdm(train_loader), 0):

#     # zero_grad
#     for param in model_restoration.parameters():
#         param.grad = None


#     target = data[0].cuda()
#     input_ = data[1].cuda()

#     restored = model_restoration(input_)

#     loss=criterion(output)

#     loss.backward()
#     optimizer.step()
#     epoch_loss +=loss.item()
