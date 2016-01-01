# -*- coding: utf-8 -*-
import cv2
import os
import imutils
import math
import numpy as np


#放缩比例
fg = 3

track = [] #当前轨迹集合
trcur = [] #当前目标位置集合

#基本参数集合
paras = {'mintgtarea':600, #目标最小像素数 600
         'maxdd':50,
         'maxdt':100
         }


#输出vpath下所有视频的路径到 opath目录下面
def outroutes(vpath,opath):
    print vpath
    vs = os.listdir(vpath)                                      #获取指定目录中的内容
    for v in vs:
        print v
        vn = v.split('-')                                       #根据'-'拆分v
        print int(vn[2]),int(vn[3]),int(vn[4][:-4])
        vp  = vpath+'/'+v                                       #输入的视频文件的路径+文件名
        vo  = opath+'/'+v                                       #输出的视频文件的路径

        #判断vo这个路径是否存在，若不存在则建立路径vo（在输出文件夹中建立以相应视频文件命名的文件夹）
        if not os.path.exists(vo):
            os.mkdir(vo)
            os.mkdir(vo+'/bg')                                  #在vo目录下建立bg子文件夹
            os.mkdir(vo+'/fg')                                  #在vo目录下建立fg子文件夹
            getTrack(vp,vo)                                     #调用getTrack函数

    return

#获取视频videofile中的轨迹，videofile为视频文件位置，voutpath为结果输出路径
def getTrack(videofile,voutpath):
    print videofile
    global trcur                                                #定义全局变量trcur
    global track                                                #定义全局变量track
    trcur = []
    track = []
    vdir,vname = os.path.split(videofile)                       #将videofile分割成路径和文件名的二元组返回，即vdir=视频文件的路径，vname=视频文件的文件名

    cap = cv2.VideoCapture(videofile)                           #读取视频

    i = 0                                                       #i记录帧数
    firstFrame = None                                           #初始化视频流的第一帧
    chi = i
    track = []
    bi = 0                                                      #bi记录背景序号
    bgflg = False

    while(1):

        ret ,frame = cap.read()                                 #调用cap.read（）返回一个二元组，元组的第一个值是ret，表明是否成功从缓冲中读取了frame，元组的第二个值就是frame本身
        if not ret:                                             #如果不能抓取到一帧，说明已经到了视频的结尾，跳出循环
            print 'output track:'
            outputTrack(voutpath+'/'+str(bi))
            cv2.destroyAllWindows()
            cap.release()
            return

        frame = imutils.resize(frame, width=int(400*fg))        #调整帧的大小,width是整数
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #转换为灰度图像

        i+=1
        if i%30!=0:                                             #每30帧抽一次
            continue
        if ret == True:                                         #可以抓取到一帧，说明还没有到视频的结尾

            gray = cv2.GaussianBlur(gray, (19,19), 0)           #对灰度图像进行高斯滤波，核宽和核高必须为正奇数
            if firstFrame is None:                              #如果firstframe没有初始化，我们将它保存然后继续处理视频的下一帧
                firstFrame = gray
                cv2.imwrite(voutpath+'/bg/'+str(bi)+'-'+str(i)+'.jpg',frame)            #只有第一帧的时候if语句会判断为True，所以保存的是frame就可以
                #将检测到的背景图片输出到文件（输出目录中以视频文件命名的文件夹下的bg文件夹中），文件名为背景序号（bi）与当前帧数（i）
                bi+=1
                bgflg = False
            #计算当前帧与第一帧的不同
            frameDelta = cv2.absdiff(firstFrame, gray)              #将当前帧与背景相减
            thresh = cv2.threshold(frameDelta, 36, 255, cv2.THRESH_BINARY)[1]           #小于阈值设置为黑色（0），大于阈值设置为白色（255）
            #扩展阈值图像填充孔洞，然后找到阈值图像上的轮廓
            kernel = np.ones((6,6),np.uint8)
            thresh = cv2.dilate(thresh,kernel,iterations = 4)
            cv2.imshow("Thresh", thresh)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            cv2.imshow('diff',frameDelta)
            ttarea,tt = add2track(cnts,i)                       #调用add2track（cnts,i）函数，ttarea为轮廓的总面积，tt为轮廓的位置的集合（列表）
            track,trcur,ched,addtt = exttrack(tt)               #调用exttrack（tt）函数，返回新轨迹集合，当前目标集合，轨迹集合是否有修改，addtt用于输出目标图像
            ti = 0
            for t in addtt:
                crop_img = frame[t[1]:(t[1]+t[3]),t[0]:(t[0]+t[2])]             #crop_img为在一帧中的检测出的轮廓
                cv2.imwrite(voutpath+'/fg/'+str(i)+'-'+str(ti)+'.jpg',crop_img)
                #将前景目标进行输出，输出在输出文件夹下的该视频文件夹下的fg文件夹中，输出格式为帧数（i），目标在当前场景中的序号（ti）
                ti+=1                                                           #ti为目标在当前场景中的序号
            if ched:                                                            #轨迹集合发生改变
                chi = i

            c = 0

            if ttarea>600*fg*300 or i-chi>100: #判断背景是否法生变化，如过目标面积大于600*300×fg或轨迹改变的帧差超过100，则说明背景变了
                firstFrame = gray
                bgflg = True

            elif bgflg: #如果背景有变化，则输出当前背景，输出已检测到的路径，并将背景设为未变化
                cv2.imwrite(voutpath+'/bg/'+str(bi)+'-'+str(i)+'.jpg',frame)
                outputTrack(voutpath+'/'+str(bi)) #调用outputTrack（）函数输出当前检测到的路径
                bi+=1
                bgflg = False
            for ts in track:
                for ii in range(len(ts)-1):
                    t = ts[ii]                  #起始点的坐标
                    td = ts[ii+1]               #下一个点的坐标
                    #cv2.rectangle(frame, (t[0], t[1]), (t[0] + t[2], t[1] + t[3]), (0, c, 0), 2)

                    cv2.line(frame,(t[0]+t[2]/2, t[1]+t[3]),(td[0]+td[2]/2, td[1]+td[3]),(0, c, 0),2)       #画轨迹
                c+=80                           #在当前场景中出现的序号不同画不同颜色的线,当track=[ [[],[]...[]] ,[[],[]...[]] ]时用深浅不同的颜色
            cv2.imshow('imgf',frame)

            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break
        else:
            break
    outputTrack(voutpath+'/'+str(bi+1)) #输出最后检测到的路径
    cv2.destroyAllWindows()
    cap.release()

#处理寻找到的目标cnts
def add2track(cnts,  i):
    ttarea = 0
    tt = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)              #获取长方形轮廓的位置（x，y为左下角坐标）
        area = cv2.contourArea(c)                       #计算长方形轮廓的面积
        ttarea += area
        if area >600*fg:
            tt.append([x, y, w, h, i])                  #将列表最后添加新元素，输出为 [横坐标x,纵坐标y,宽度w,高度h,帧数i]，
        #将每一帧中的轮廓坐标放到一个[]中作为tt


    return ttarea, tt #cnts总面，目标位置

#对当前track进行扩展，将新检测出的目标集合tt与当前track进行合并
def exttrack(tt):
    tt = gl(tt)                                 #调用gl（tt）函数除去时间数字与较小目标
    ched = False                                #标记轨迹是否发生改变
    addtt = []
    i = 0
    global trcur
    global track
    for t in tt:                                #对一帧的tt循环
        if len(t)==0:
            continue

        if t[1]<20*fg and t[1]+t[3]<50*fg:
            continue
        k = select(t,trcur)                     #调用select（t，trcur）从trcur中选择与t最接近的一个目标，k为trcur的序号
        ched = True
        t.append(i) #i为当前目标编号
        i+=1
        addtt.append(t)                         #将t添加到addtt列表的最后
        if k!=-1:                               #在trcur中找到一个与t最接近的目标
            if trcur[k][0]==t[0] and trcur[k][1]==t[1] and trcur[k][2]==t[2] and trcur[k][3]==t[3]:
                ched = False                    #如果trcur中找到的目标与t相等，则没有发生改变
            track[k].append(t)                  #将t添加到目标集合track[k]中,eg t=[466, 1, 209, 139, 1860, 1]
            trcur[k] = t                        #用t替代trcur[k]
        else:                                   #在trcur中没有找到与t接近的目标
            track.append([t])                   #将[t]添加到track列表的最后，eg [t]=[[466, 1, 209, 139, 1860, 1]]
            trcur.append(t)                     #将t添加到trcur列表的最后
    #返回 新轨迹集合，当前目标集合，轨迹集合是否有修改，addtt用于输出目标图像
    return track,trcur,ched,addtt

#对识别的目标进行过滤，如果目标位于上部的时间区域，则对其进行去除，或者目标小于100像素，也进行过滤
def gl(tt):
    res = []
    for t in tt:                                        #tt为轮廓的位置的集合（列表）
        if len(t)==0:
            continue
        if t[1]<20*fg and t[1]+t[3]<50*fg:
            continue                                    #如果轮廓出现在时间的位置则忽略
        if t[3]*t[2]<100*fg*fg:
            continue                                    #如果轮廓的大小小于100*fg*fg则忽略
        res.append(t)                                   #将t添加到res（列表）中，res中为经过筛选的轮廓的位置的集合
    return  res

#从ts中选择与t最接近的一个目标
def select(t,ts):

    k = -1
    min = 1000000

    for i in range(len(ts)):
        da = getdiff(t,ts[i])                           #调用getdiff（t,ts[i]）函数，计算t与ts[i]的面积差与距离差的平方的加权值
        if da<min:
            min = da                                    #更新最小距离
            k = i                                       #记录接近的是ts中的第几个目标
    return k

# 计算t1与t2的差距
def getdiff(t1,t2):
    c1x,c1y = getCen(t1)                                    #调用getCen(t1)函数计算矩形t1 [x,y,w,h] 的中心坐标
    c2x,c2y = getCen(t2)
    dx = c1x-c2x
    dy = c1y-c2y
    dw = t1[2]-t2[2]
    dh = t1[3]-t2[3]

    dd = math.sqrt(dx*dx+dy*dy) #t1与t2中心之间的距离
    da = t1[2]*t1[3]-t2[2]*t2[3] #t1与t2面积之差
    dt = t1[4] - t2[4] #t1与t2的帧差
    if dd>50*fg or abs(dt)>100:
        return 1000000 #如果距离大于50×fg或者帧差大于100则返回一个大的距离
    else:
        return abs(da)+dd*dd #否则返回一个面积差与距离差平方的平均加权值

# 获取矩形t1 [x,y,w,h] 的中心坐标
def getCen(t1):
    c1x = t1[0]+int(t1[2]/2)
    c1y = t1[1]+int(t1[3]/2)
    return c1x,c1y

#输出轨迹到bi文件
def outputTrack(bi):
    global trcur
    global track
    fl = bi
    f = open(fl, 'w+')

    for ts in track:

        print >>f, ts
    f.close()
    trcur = []
    track = []
    print 'output'


#本地测试文件
if __name__ == '__main__':
    vpath = '/home/zhangjianshu/桌面/test videotrack'           #输入文件的路径
    opath = '/home/zhangjianshu/桌面/result'                    #输出文件的路径

    outroutes(vpath,opath)


