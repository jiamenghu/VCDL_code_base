python main.py -q 37 --GPU 3 --bs 32  --ps 64  --lr 1e-2  --lr_min 1e-6 --trainset BVIDVC --stride_frame 50  --model ./experiment_lr/model/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_37.pth >./experiment_lr/log/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_37.log 
python main.py -q 32 --GPU 3 --bs 32  --ps 64  --lr 1e-2  --lr_min 1e-6 --trainset BVIDVC --stride_frame 50  --model ./experiment_lr/model/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_32.pth >./experiment_lr/log/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_32.log 
python main.py -q 27 --GPU 3 --bs 32  --ps 64  --lr 1e-2  --lr_min 1e-6 --trainset BVIDVC --stride_frame 50  --model ./experiment_lr/model/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_27.pth >./experiment_lr/log/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_27.log 
python main.py -q 22 --GPU 3 --bs 32  --ps 64  --lr 1e-2  --lr_min 1e-6 --trainset BVIDVC --stride_frame 50  --model ./experiment_lr/model/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_22.pth >./experiment_lr/log/ConvBNNet_10x1_P64_50BVIDVC_lr12_16_22.log 



python main.py -q 37 --GPU 7 --bs 1  --ps 16   --trainset BVIDVC --stride_frame 10  --model ./model/test_ConvBNNet_10x1_P64_50BVIDVC_lr12_16_37.pth >test_ConvBNNet_10x1_P64_50BVIDVC_lr12_16_37.log 


