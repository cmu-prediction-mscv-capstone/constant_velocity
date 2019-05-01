import os
import numpy as np
#from utils import DataLoader
import cv2
import matplotlib.pyplot as plt
import glob
folder="datasets_sgan"
dataset="eth/test"
file_path=os.path.join(folder,dataset)
file_name=glob.glob(file_path+"/*.txt")[0]
print(file_name)
file_pointer=open(file_name)
lines=file_pointer.readlines()
all_lines=[]
for line in lines:
    split_line=line.split("\t")
    all_lines.append([int(float(split_line[0])),int(float(split_line[1])),float(split_line[2]),float(split_line[3])])
frame=all_lines[0][0]
frame_list=[[all_lines[0]]]
#print(frame_list)
for line in all_lines:
    curr_frame=line[0]
    prev_frame=frame_list[-1][0][0]
    #print(prev_frame,curr_frame,prev_frame==curr_frame)
    if prev_frame==curr_frame:
        #print("here")
        frame_list[-1].append(line)
    else:
        frame_list.append([line])
count=0
error=0
error_fde=0
count_fde=0
step_x=2
step_y=12
increment_xy=step_x
x_frame_start=0
y_frame_start=x_frame_start+step_x
y_frame_end=x_frame_start+step_x+step_y
print("The length of frame list is ",len(frame_list))
while(y_frame_end<=len(frame_list)):
    x=frame_list[x_frame_start:y_frame_start]
    y=frame_list[y_frame_start:y_frame_end]
    #print(x_frame_start,y_frame_start,y_frame_end)
    for ped_loop_2 in  x[-1]:
        for ped_loop_1 in x[-2]:
            if(ped_loop_1[1]==ped_loop_2[1]):
                for frame_number in range(0,step_y):
                    x_pred=ped_loop_2[2]+ (ped_loop_2[2]-ped_loop_1[2])*(frame_number+1)
                    y_pred=ped_loop_2[3]+ (ped_loop_2[3]-ped_loop_1[3])*(frame_number+1)
                    for ped_loop_3 in frame_list[y_frame_start+frame_number]:
                        if ped_loop_1[1]==ped_loop_3[1]:
                            error+=np.linalg.norm([ped_loop_3[2]-x_pred,ped_loop_3[3]-y_pred])
                            count+=1
                            if(frame_number==step_y-1):
                                error_fde+=np.linalg.norm([ped_loop_3[2]-x_pred,ped_loop_3[3]-y_pred])
                                count_fde+=1

    x_frame_start+=step_x
    y_frame_start=x_frame_start+step_y
    y_frame_end=y_frame_start+step_y
print("The average displacement error for seq_length {} and pred_length {} is {} for count {}".format(step_x,step_y,error/count,count))
print("The average final displacement error for seq_length {} and pred_length {} is {} for count {}".format(step_x,step_y,error_fde/count_fde,count_fde))
