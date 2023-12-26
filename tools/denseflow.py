import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
#from IPython import embed #to debug
import imageio

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    # height = flow.shape[0]
    # width  = flow.shape[1]
    # resized_flow = cv2.resize(flow, (width//4, height//4))
    return flow

def save_flows(flows,image,save_dir,num,bound,video_name,data_root,new_dir):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    flow_zero = np.zeros((flow_x.shape[0],flow_x.shape[1],3), dtype=np.uint8)
    flow_zero[:, :, 0] = flow_x
    flow_zero[:, :, 1] = flow_y
    flow_zero[:, :, 2] = flow_x
    flow_uint8 = flow_zero.astype(np.uint8)
    save_dir = save_dir.split("_")[0]
    save_name = video_name.split(".")[0]
    if not os.path.exists(os.path.join(data_root, new_dir, save_dir, save_name)):
        os.makedirs(os.path.join(data_root, new_dir, save_dir, save_name))
    #save the image
    save_zero = os.path.join(data_root, new_dir, save_dir, save_name, 'flow_{:03d}.jpg'.format(num))
    imageio.imwrite(save_zero, flow_uint8)
    save_img=os.path.join(data_root,new_dir,save_dir, save_name,'rgb_{:03d}.jpg'.format(num))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_uint8 = image.astype(np.uint8)
    imageio.imwrite(save_img,image_uint8)
    #save the flows
    # save_x=os.path.join(data_root,new_dir,save_dir,'flow_x_{}_{:03d}.jpg'.format(save_name,num))
    # save_y=os.path.join(data_root,new_dir,save_dir,'flow_y_{}_{:03d}.jpg'.format(save_name,num))
    # # flow_x_img=Image.fromarray(flow_x)
    # # flow_y_img=Image.fromarray(flow_y)
    # # imageio.imsave(save_x,flow_x_img)
    # # imageio.imsave(save_y,flow_y_img)
    # flow_x_uint8 = flow_x.astype(np.uint8)
    # flow_y_uint8 = flow_y.astype(np.uint8)
    # imageio.imwrite(save_x, flow_x_uint8)
    # imageio.imwrite(save_y, flow_y_uint8)
    # print(save_x, save_y)
    return 0

def dense_flow(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name,save_dir,step,bound,data_root,new_dir=augs
    video_path=os.path.join(data_root,"CSL-1000",video_name.split('_')[0],video_name)
    # provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support

    # videocapture=cv2.VideoCapture(video_path)
    # if not videocapture.isOpened():
    #     print 'Could not initialize capturing! ', video_name
    #     exit()
    try:
        videocapture=cv2.VideoCapture(video_path)
    except:
        print('{} read error! '.format(video_name))
        return 0
    # if extract nothing, exit!
    if not videocapture.isOpened():
        print('Could not initialize capturing',video_path)
        exit()
    len_frame=int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 10
    end_frame = len_frame-10
    # if len_frame <= 32:
    #     start_frame = 0
    #     end_frame = len_frame - 1
    # else:
    #     start_frame = len_frame//2 - 16
    #     end_frame = len_frame//2 + 16
    frame_num=0
    image,prev_image,gray,prev_gray=None,None,None,None
    num0=start_frame
    while True:
        #frame=videocapture.read()
        if num0>end_frame:
            break
        ret, frame=videocapture.read()
        num0+=1
        if frame_num==0:
            image=np.zeros_like(frame)
            gray=np.zeros_like(frame)
            prev_gray=np.zeros_like(frame)
            prev_image=frame
            prev_gray=cv2.cvtColor(prev_image,cv2.COLOR_RGB2GRAY)
            frame_num+=1
            # to pass the out of stepped frames
            step_t=step
            while step_t>1:
                #frame=videocapture.read()
                num0+=1
                step_t-=1
            continue

        image=frame
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        frame_0=prev_gray
        frame_1=gray
        flow = cv2.calcOpticalFlowFarneback(frame_0, frame_1, None, pyr_scale=0.5, levels=3, winsize=30, iterations=3,
                                            poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # ##default choose the tvl1 algorithm
        # dtvl1=cv2.optflow_DualTVL1OpticalFlow.create()
        # # dtvl1 = cv2.calcOpticalFlowFarneback()
        # flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        save_flows(flow,image,save_dir,frame_num,bound,video_name,data_root,new_dir) #this is to save flows and img.
        prev_gray=gray
        prev_image=image

        frame_num+=1
        # to pass the out of stepped frames
        step_t=step
        while step_t>1:
            #frame=videocapture.read()
            num0+=1
            step_t-=1
    print("save:::::::", video_path)


def get_video_list():
    video_list=[]
    for cls_names in os.listdir(videos_root):
        cls_path=os.path.join(videos_root,cls_names)
        for video_ in os.listdir(cls_path):
            video_list.append(video_)
    video_list.sort()
    return video_list,len(video_list)



def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset',default='CSL-1000',type=str,help='set the dataset name, to find the data path')
    parser.add_argument('--data_root',default=r'D:\code',type=str)   #F:/uestc/code/dataset  D:/Desktop/server
    parser.add_argument('--new_dir',default='CSL-flow',type=str)
    parser.add_argument('--num_workers',default=4,type=int,help='num of workers to act multi-process')
    parser.add_argument('--step',default=1,type=int,help='gap frames')
    parser.add_argument('--bound',default=15,type=int,help='set the maximum of optical flow')
    parser.add_argument('--s_',default=0,type=int,help='start id')
    parser.add_argument('--e_',default=13320,type=int,help='end id')
    parser.add_argument('--mode',default='run',type=str,help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    # example: if the data path not setted from args,just manually set them as belows.
    #dataset='ucf101'
    #data_root='/S2/MI/zqj/video_classification/data'
    #data_root=os.path.join(data_root,dataset)

    args=parse_args()
    data_root = args.data_root
    # data_root=os.path.join(args.data_root,args.dataset)
    videos_root=os.path.join(args.data_root,args.dataset)

    #specify the augments
    num_workers=args.num_workers
    step=args.step
    bound=args.bound
    s_=args.s_
    e_=args.e_
    new_dir=args.new_dir
    mode=args.mode

    #get video list
    video_list,len_videos=get_video_list()  #name
    video_list=video_list[s_:e_]
    # len_videos=min(e_-s_,13320-s_) # if we choose the ucf101
    print('find {} videos.'.format(len_videos))
    flows_dirs=[video.split('.')[0] for video in video_list]
    print('get videos list done! ')
    pool=Pool(num_workers)
    if mode=='run':
        pool.map(dense_flow,zip(video_list,flows_dirs,[step]*len(video_list),[bound]*len(video_list),[data_root]*len(video_list),[new_dir]*len(video_list)))
    else: #mode=='debug
        dense_flow((video_list[0],flows_dirs[0],step,bound,[data_root]*len(video_list),[new_dir]*len(video_list)))
