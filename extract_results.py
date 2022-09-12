import os
#import xlwt
import math
import argparse 

def load_file_list(directory):
    
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        if filename.split('_')[-1]=='22.txt':
          list.append(filename.split('AI_')[0])
    return sorted(list)

p=argparse.ArgumentParser()
p.add_argument('-v',help='VTM or HM default:HM',required =False)

p.add_argument('-t',help='txt文件路径',required =False)
p.add_argument('-o',help='统计输出excel名字',required =False)

args = p.parse_args()
if args.t:
    txt_path=args.t
if args.o:
    excel_path = args.o
data=[]
with open(txt_path,'r') as ftxt:
        content = ftxt.read().splitlines()
        flag=0
        for line in content:
            if line=='&& Start TEST':
                flag=1
                continue
            if line=='test CTU':
                break
            if flag==1:
                text=line.split(':')
                if len(text)<5:
                    continue
                data.append([text[0],float(text[2].split(',')[0]),float(text[3].split(',')[0]),float(text[4].split(',')[0]),float(text[5].split(',')[0]),float(text[6].split(',')[0]),float(text[7].split('}')[0])])
print(data)

for i in data:
    print(i[0])

# for ypsnr in data:
#     print('%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t'%(round(ypsnr[1],4),round(ypsnr[2],4),round(ypsnr[3],4),round(ypsnr[4],4),round(ypsnr[5],4),round(ypsnr[6],4)))

for ypsnr in data:
    print('%0.4f'%(round(ypsnr[6],4)))