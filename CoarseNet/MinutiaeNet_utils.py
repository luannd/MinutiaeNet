import os

import glob
import shutil
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc, signal, spatial
from skimage.filters import gaussian
import cv2
import math

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def re_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def init_log(output_dir):
    re_mkdir(output_dir)
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        filename=os.path.join(output_dir, 'log.log'),
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

def copy_file(path_s, path_t):
    shutil.copy(path_s, path_t)   

def get_files_in_folder(folder, file_ext=None):
    files = glob.glob(os.path.join(folder, "*" + file_ext))
    files_name = []
    for i in files:
        _, name = os.path.split(i)
        name, ext = os.path.splitext(name)
        files_name.append(name)
    return np.asarray(files), np.asarray(files_name)

def point_rot(points, theta, b_size, a_size):
    cosA = np.cos(theta)
    sinA = np.sin(theta)
    b_center = [b_size[1]/2.0, b_size[0]/2.0]
    a_center = [a_size[1]/2.0, a_size[0]/2.0]
    points = np.dot(points-b_center, np.array([[cosA,-sinA],[sinA,cosA]]))+a_center
    return points

def mnt_reader(file_name):
    f = open(file_name)
    minutiae = []
    for i, line in enumerate(f):
        if i < 4 or len(line) == 0: continue
        w, h, o = [float(x) for x in line.split()]
        w, h = int(round(w)), int(round(h))
        minutiae.append([w, h, o])
    f.close()
    return minutiae

def mnt_writer(mnt, image_name, image_size, file_name):
    f = open(file_name, 'w')
    f.write('%s\n'%(image_name))
    f.write('%d %d %d\n'%(mnt.shape[0], image_size[0], image_size[1]))
    for i in xrange(mnt.shape[0]):
        f.write('%d %d %.6f %.4f\n'%(mnt[i,0], mnt[i,1], mnt[i,2], mnt[i,3]))
    f.close()
    return

def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    # Bounding box
    nstds = 3
    xmax = ksize[0]/2
    ymax = ksize[1]/2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_cos = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb_sin = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    return gb_cos, gb_sin

def gabor_bank(stride=2,Lambda=8):

    filters_cos = np.ones([25,25,180/stride], dtype=float)
    filters_sin = np.ones([25,25,180/stride], dtype=float)

    for n, i in enumerate(xrange(-90,90,stride)):
        theta = i*np.pi/180.
        kernel_cos, kernel_sin = gabor_fn((24,24),4.5, -theta, Lambda, 0, 0.5)
        filters_cos[..., n] = kernel_cos
        filters_sin[..., n] = kernel_sin

    filters_cos = np.reshape(filters_cos,[25,25,1,-1])
    filters_sin = np.reshape(filters_sin,[25,25,1,-1])
    return filters_cos, filters_sin

def gaussian2d(shape=(5,5),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gausslabel(length=180, stride=2):
    gaussian_pdf = signal.gaussian(length+1, 3)
    label = np.reshape(np.arange(stride/2, length, stride), [1,1,-1,1])
    y = np.reshape(np.arange(stride/2, length, stride), [1,1,1,-1])
    delta = np.array(np.abs(label - y), dtype=int)
    delta = np.minimum(delta, length-delta)+length/2
    return gaussian_pdf[delta]

def angle_delta(A, B, max_D=np.pi*2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D-delta)
    return delta
def fmeasure(P, R):
    return 2*P*R/(P+R+1e-10)
def distance(y_true, y_pred, max_D=16, max_O=np.pi/6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    O = spatial.distance.cdist(np.reshape(y_true[:, 2], [-1, 1]), np.reshape(y_pred[:, 2], [-1, 1]), angle_delta)
    return (D<=max_D)*(O<=max_O)

def metric_P_R_F(y_true, y_pred, maxd=16, maxo=np.pi/6):
    # Calculate Precision, Recall, F-score
    if y_pred.shape[0]==0 or y_true.shape[0]==0:
        return 0,0,0,0,0

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    total_gt, total = float(y_true.shape[0]), float(y_pred.shape[0])
    # Using L2 loss
    dis = spatial.distance.cdist(y_pred[:, :2], y_true[:, :2], 'euclidean')
    mindis,idx = dis.min(axis=1),dis.argmin(axis=1)

    #Change to adapt to new annotation: old version. When training, comment it
    # y_pred[:,2] = -y_pred[:,2]

    angle = abs(np.mod(y_pred[:,2],2*np.pi) - y_true[idx,2])
    angle = np.asarray([angle, 2*np.pi-angle]).min(axis=0)

    # Satisfy the threshold
    tmp=(mindis <= maxd) & (angle<=maxo)
    #print('mindis,idx,angle,tmp=%s,%s,%s,%s'%(mindis,idx,angle,tmp))

    precision = len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)]))/float(y_pred.shape[0])
    recall = len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)]))/float(y_true.shape[0])
    #print('pre=%f/ %f'%(len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)])),float(y_pred.shape[0])))
    #print('recall=%f/ %f'%(len(np.unique(idx[(mindis <= maxd) & (angle<=maxo)])),float(y_true.shape[0])))
    if recall!=0:
        loc = np.mean(mindis[(mindis <= maxd) & (angle<=maxo)])
        ori = np.mean(angle[(mindis <= maxd) & (angle<=maxo)])
    else:
        loc = 0
        ori = 0
    return precision, recall, fmeasure(precision, recall), loc, ori

def nms(mnt):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi/6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in xrange(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]

def fuse_nms(mnt, mnt_set_2):
    if mnt.shape[0]==0:
        return mnt
    # sort score
    all_mnt = np.concatenate((mnt, mnt_set_2))

    mnt_sort = all_mnt.tolist()
    mnt_sort.sort(key=lambda x:x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    # cal distance
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=2*np.pi).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in xrange(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i+1:] = keep_list[i+1:]*(1-inrange[i, i+1:])
    return mnt_sort[keep_list.astype(np.bool), :]


def py_cpu_nms(det, thresh):
    if det.shape[0]==0:
        return det
    dets = det.tolist()
    dets.sort(key=lambda x:x[3], reverse=True)
    dets = np.array(dets)

    box_sz = 25
    x1 = np.reshape(dets[:,0],[-1,1]) -box_sz
    y1 = np.reshape(dets[:,1],[-1,1]) -box_sz
    x2 = np.reshape(dets[:,0],[-1,1]) +box_sz
    y2 = np.reshape(dets[:,1],[-1,1]) +box_sz
    scores = dets[:, 2]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]

def draw_minutiae(image, minutiae, fname, saveimage= False, r=15, drawScore=False):
    image = np.squeeze(image)
    fig = plt.figure()
    

    plt.imshow(image,cmap='gray')
    plt.hold(True)
    # Check if no minutiae
    if minutiae.shape[0] > 0:
        plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
        for x, y, o, s in minutiae:
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
            if drawScore == True:
                plt.text(x - 10, y - 10, '%.2f' % s, color='yellow', fontsize=4)

    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')
    if saveimage:
        plt.savefig(fname, dpi=500, bbox_inches='tight', pad_inches = 0)
        plt.close(fig)
    else:
        plt.show()
    return

def draw_minutiae_overlay(image, minutiae, mnt_gt, fname, saveimage= False, r=15, drawScore=False):
    image = np.squeeze(image)
    fig = plt.figure()
    

    plt.imshow(image,cmap='gray')
    plt.hold(True)

    if mnt_gt.shape[1] > 3:
        mnt_gt = mnt_gt[:,:3]

    if mnt_gt.shape[0] > 0:
        if mnt_gt.shape[1] > 3:
            mnt_gt = mnt_gt[:, :3]
        plt.plot(mnt_gt[:, 0], mnt_gt[:, 1], 'bs', fillstyle='none', linewidth=1)
        for x, y, o in mnt_gt:
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'b-')

    if minutiae.shape[0] > 0:
        plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
        for x, y, o in minutiae:
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
            if drawScore == True:
                plt.text(x - 10, y - 10, '%.2f' % s, color='yellow', fontsize=4)

    plt.axis([0,image.shape[1],image.shape[0],0])
    plt.axis('off')
    plt.show()
    if saveimage:
        plt.savefig(fname, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return


def draw_minutiae_overlay_with_score(image, minutiae, mnt_gt, fname, saveimage=False, r=15):
    image = np.squeeze(image)
    fig = plt.figure()

    plt.imshow(image, cmap='gray')
    plt.hold(True)


    if mnt_gt.shape[0] > 0:
        plt.plot(mnt_gt[:, 0], mnt_gt[:, 1], 'bs', fillstyle='none', linewidth=1)
        if mnt_gt.shape[1] > 3:
            for x, y, o, s in mnt_gt:
                plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'b-')
                plt.text(x - 10, y - 5, '%.2f' % s, color='green', fontsize=4)
        else:
            for x, y, o in mnt_gt:
                plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'b-')

    if minutiae.shape[0] > 0:
        plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
        for x, y, o, s in minutiae:
            plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'r-')
            plt.text(x-10,y-10,'%.2f'%s,color='yellow',fontsize=4)

    plt.axis([0, image.shape[1], image.shape[0], 0])
    plt.axis('off')

    if saveimage:
        plt.savefig(fname, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return

def draw_ori_on_img(img, ori, mask, fname, saveimage=False, coh=None, stride=16):
    ori = np.squeeze(ori)
    #mask = np.squeeze(np.round(mask))

    img = np.squeeze(img)
    ori = ndimage.zoom(ori, np.array(img.shape)/np.array(ori.shape, dtype=float), order=0)
    if mask.shape != img.shape:
        mask = ndimage.zoom(mask, np.array(img.shape)/np.array(mask.shape, dtype=float), order=0)
    if coh is None:
        coh = np.ones_like(img)
    fig = plt.figure()
    plt.imshow(img,cmap='gray')
    plt.hold(True)  
    for i in xrange(stride,img.shape[0],stride):
        for j in xrange(stride,img.shape[1],stride):
            if mask[i, j] == 0:
                continue
            x, y, o, r = j, i, ori[i,j], coh[i,j]*(stride*0.9)
            plt.plot([x, x+r*np.cos(o)], [y, y+r*np.sin(o)], 'r-')
    plt.axis([0,img.shape[1],img.shape[0],0])
    plt.axis('off')
    if saveimage:
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return

def local_constrast_enhancement(img):
    img = img.astype(np.float32)
    meanV = cv2.blur(img,(15,15))
    normalized = img - meanV
    var = abs(normalized)

    var = cv2.blur(var,(15,15))
    normalized = normalized/(var+10) *0.75
    normalized = np.clip(normalized, -1, 1)
    normalized = (normalized+1)*127.5
    return normalized

def get_quality_map_ori_dict(img, dict, spacing, dir_map = None, block_size = 16):
    if img.dtype=='uint8':
        img = img.astype(np.float)
    img = FastEnhanceTexture(img)
    h, w = img.shape
    blkH, blkW = dir_map.shape

    quality_map = np.zeros((blkH,blkW),dtype=np.float)
    fre_map = np.zeros((blkH,blkW),dtype=np.float)
    ori_num = len(dict)
    #dir_map = math.pi/2 - dir_map
    dir_ind = dir_map*ori_num/math.pi
    dir_ind = dir_ind.astype(np.int)
    dir_ind = dir_ind%ori_num

    patch_size = np.sqrt(dict[0].shape[1])
    patch_size = patch_size.astype(np.int)
    pad_size = (patch_size-block_size)//2
    img = np.lib.pad(img, (pad_size, pad_size), 'symmetric')
    for i in range(0,blkH):
        for j in range(0,blkW):
            ind = dir_ind[i,j]
            patch = img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size]

            patch = patch.reshape(patch_size*patch_size,)
            patch = patch - np.mean(patch)
            patch = patch / (np.linalg.norm(patch)+0.0001)
            patch[patch>0.05] = 0.05
            patch[patch<-0.05] = -0.05

            simi = np.dot(dict[ind], patch)
            similar_ind = np.argmax(abs(simi))
            quality_map[i,j] = np.max(abs(simi))
            fre_map[i,j] = 1./spacing[ind][similar_ind]

    quality_map = gaussian(quality_map,sigma=2)
    return quality_map, fre_map

def FastEnhanceTexture(img,sigma=2.5,show=False):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(-FFTsize / 2, FFTsize / 2), range(-FFTsize / 2, FFTsize / 2))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r/FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma)** 4)
    img_low = LowpassFiltering(img, L)

    gradim1=  compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1,L)

    gradim2=  compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2,L)

    diff = gradim1-gradim2
    ar1 = np.abs(gradim1)
    diff[ar1>1] = diff[ar1>1]/ar1[ar1>1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff-cmin)/(cmax-cmin)
    weight[diff<cmin] = 0
    weight[diff>cmax] = 1


    u = weight * img_low + (1-weight)* img

    temp = img - u

    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)

    temp1[temp1 < 0] = 0
    temp1[temp1 >255] = 255
    v = temp1
    if show:
        plt.imshow(v,cmap='gray')
        plt.show()
    return v

def compute_gradient_norm(input):
    input = input.astype(np.float32)

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out

def LowpassFiltering(img,L):
    h,w = img.shape
    h2,w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2-h, 0, w2-w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L
    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h,:w]

    return rec_img

def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))

def construct_dictionary(ori_num = 30):
    ori_dict = []
    s = []
    for i in range(ori_num):
        ori_dict.append([])
        s.append([])

    patch_size2 = 16
    patch_size = 32
    dict_all = []
    spacing_all = []
    ori_all = []
    Y, X = np.meshgrid(range(-patch_size2,patch_size2), range(-patch_size2,patch_size2))

    for spacing in range(6,13):
        for valley_spacing in range(3,spacing//2):
            ridge_spacing = spacing - valley_spacing
            for k in range(ori_num):
                theta = np.pi/2-k*np.pi / ori_num
                X_r = X * np.cos(theta) - Y * np.sin(theta)
                for offset in range(0,spacing-1,2):
                    X_r_offset = X_r + offset + ridge_spacing / 2
                    X_r_offset = np.remainder(X_r_offset, spacing)
                    Y1 = np.zeros((patch_size, patch_size))
                    Y2 = np.zeros((patch_size, patch_size))
                    Y1[X_r_offset <= ridge_spacing] = X_r_offset[X_r_offset <= ridge_spacing]
                    Y2[X_r_offset > ridge_spacing] = X_r_offset[X_r_offset > ridge_spacing] - ridge_spacing
                    element = -np.sin(2 * math.pi * (Y1 / ridge_spacing / 2)) + np.sin(2 * math.pi * (Y2 / valley_spacing / 2))

                    element = element.reshape(patch_size*patch_size,)
                    element = element-np.mean(element)
                    element  = element/ np.linalg.norm(element)
                    ori_dict[k].append(element)
                    s[k].append(spacing)
                    dict_all.append(element)
                    spacing_all.append(1.0/spacing)
                    ori_all.append(theta)

    for i in range(len(ori_dict)):
        ori_dict[i] = np.asarray(ori_dict[i])
        s[k] = np.asarray(s[k])

    dict_all = np.asarray(dict_all)
    dict_all = np.transpose(dict_all)
    spacing_all = np.asarray(spacing_all)
    ori_all = np.asarray(ori_all)


    return ori_dict, s, dict_all, ori_all,spacing_all

def get_maps_STFT(img,patch_size = 64,block_size = 16, preprocess = False):
    assert len(img.shape) == 2

    nrof_dirs = 16
    ovp_size = (patch_size-block_size)//2
    if preprocess:
        img = FastEnhanceTexture(img, sigma=2.5, show=False)

    img = np.lib.pad(img, (ovp_size,ovp_size),'symmetric')
    h,w = img.shape
    blkH = (h - patch_size)//block_size+1
    blkW = (w - patch_size)//block_size+1
    local_info = np.empty((blkH,blkW),dtype = object)

    x, y = np.meshgrid(range(-patch_size / 2,patch_size / 2), range(-patch_size / 2,patch_size / 2))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    r = np.sqrt(x*x + y*y) + 0.0001


    RMIN = 3  # min allowable ridge spacing
    RMAX = 18 # maximum allowable ridge spacing
    FLOW = patch_size / RMAX
    FHIGH = patch_size / RMIN
    dRLow = 1. / (1 + (r / FHIGH) ** 4)
    dRHigh = 1. / (1 + (FLOW / r) ** 4)
    dBPass = dRLow * dRHigh  # bandpass

    dir = np.arctan2(y,x)
    dir[dir<0] = dir[dir<0] + math.pi
    dir_ind = np.floor(dir/(math.pi/nrof_dirs))
    dir_ind = dir_ind.astype(np.int,copy=False)
    dir_ind[dir_ind==nrof_dirs] = 0


    dir_ind_list = []
    for i in range(nrof_dirs):
        tmp = np.argwhere(dir_ind == i)
        dir_ind_list.append(tmp)


    sigma = patch_size/3
    weight = np.exp(-(x*x + y*y)/(sigma*sigma))


    for i in range(0,blkH):
        for j in range(0,blkW):
            patch =img[i*block_size:i*block_size+patch_size,j*block_size:j*block_size+patch_size].copy()
            local_info[i,j] = local_STFT(patch,weight,dBPass)
            local_info[i, j].analysis(r,dir_ind_list)


    # get the ridge flow from the local information
    dir_map,fre_map = get_ridge_flow_top(local_info)
    dir_map = smooth_dir_map(dir_map)

    return dir_map, fre_map

def smooth_dir_map(dir_map,sigma=2.0,mask = None):

    cos2Theta = np.cos(dir_map * 2)
    sin2Theta = np.sin(dir_map * 2)
    if mask is not None:
        assert (dir_map.shape[0] == mask.shape[0])
        assert (dir_map.shape[1] == mask.shape[1])
        cos2Theta[mask == 0] = 0
        sin2Theta[mask == 0] = 0

    cos2Theta = gaussian(cos2Theta, sigma, multichannel=False, mode='reflect')
    sin2Theta = gaussian(sin2Theta, sigma, multichannel=False, mode='reflect')

    dir_map = np.arctan2(sin2Theta,cos2Theta)*0.5


    return dir_map

def get_ridge_flow_top(local_info):

    blkH,blkW = local_info.shape
    dir_map = np.zeros((blkH,blkW)) - 10
    fre_map = np.zeros((blkH, blkW)) - 10
    for i in range(blkH):
        for j in range(blkW):
            if local_info[i,j].ori is None:
                continue

            dir_map[i,j] = local_info[i,j].ori[0] #+ math.pi*0.5
            fre_map[i,j] = local_info[i,j].fre[0]
    return dir_map,fre_map


class local_STFT:
    def __init__(self,patch,weight = None, dBPass = None):


        if weight is not None:
            patch = patch * weight
        patch = patch - np.mean(patch)
        norm = np.linalg.norm(patch)
        patch = patch / (norm+0.000001)

        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        if dBPass is not None:
            fshift = dBPass * fshift

        self.patch_FFT = fshift
        self.patch = patch
        self.ori = None
        self.fre = None
        self.confidence = None
        self.patch_size = patch.shape[0]

    def analysis(self,r,dir_ind_list=None,N=2):

        assert(dir_ind_list is not None)
        energy = np.abs(self.patch_FFT)
        energy = energy / (np.sum(energy)+0.00001)
        nrof_dirs = len(dir_ind_list)

        ori_interval = math.pi/nrof_dirs
        ori_interval2 = ori_interval/2


        pad_size = 1
        dir_norm = np.zeros((nrof_dirs + 2,))
        for i in range(nrof_dirs):
            tmp = energy[dir_ind_list[i][:, 0], dir_ind_list[i][:, 1]]
            dir_norm[i + 1] = np.sum(tmp)

        dir_norm[0] = dir_norm[nrof_dirs]
        dir_norm[nrof_dirs + 1] = dir_norm[1]

        # smooth dir_norm
        smoothed_dir_norm = dir_norm
        for i in range(1, nrof_dirs + 1):
            smoothed_dir_norm[i] = (dir_norm[i - 1] + dir_norm[i] * 4 + dir_norm[i + 1]) / 6

        smoothed_dir_norm[0] = smoothed_dir_norm[nrof_dirs]
        smoothed_dir_norm[nrof_dirs + 1] = smoothed_dir_norm[1]

        den = np.sum(smoothed_dir_norm[1:nrof_dirs + 1]) + 0.00001  # verify if den == 1
        smoothed_dir_norm = smoothed_dir_norm/den  # normalization if den == 1, this line can be removed

        ori = []
        fre = []
        confidence = []

        wenergy = energy*r
        for i in range(1, nrof_dirs+1):
            if smoothed_dir_norm[i] > smoothed_dir_norm[i-1] and smoothed_dir_norm[i] > smoothed_dir_norm[i+1]:
                tmp_ori = (i-pad_size)*ori_interval + ori_interval2 + math.pi/2
                ori.append(tmp_ori)
                confidence.append(smoothed_dir_norm[i])
                tmp_fre = np.sum(wenergy[dir_ind_list[i-pad_size][:, 0], dir_ind_list[i-pad_size][:, 1]])/dir_norm[i]
                tmp_fre = 1/(tmp_fre+0.00001)
                fre.append(tmp_fre)


        if len(confidence)>0:
            confidence = np.asarray(confidence)
            fre = np.asarray(fre)
            ori = np.asarray(ori)
            ind = confidence.argsort()[::-1]
            confidence = confidence[ind]
            fre = fre[ind]
            ori = ori[ind]
            if len(confidence) >= 2 and confidence[0]/confidence[1]>2.0:

                self.ori = [ori[0]]
                self.fre = [fre[0]]
                self.confidence = [confidence[0]]
            elif len(confidence)>N:
                fre = fre[:N]
                ori = ori[:N]
                confidence = confidence[:N]
                self.ori = ori
                self.fre = fre
                self.confidence = confidence
            else:
                self.ori = ori
                self.fre = fre
                self.confidence = confidence

    def get_features_of_topN(self,N=2):
        if self.confidence is None:
            self.border_wave = None
            return
        candi_num = len(self.ori)
        candi_num = np.min([candi_num,N])
        patch_size = self.patch_FFT.shape
        for i in range(candi_num):

            kernel = gabor_kernel(self.fre[i], theta=self.ori[i], sigma_x=10, sigma_y=10)

            kernel_f = np.fft.fft2(kernel.real, patch_size)
            kernel_f = np.fft.fftshift(kernel_f)
            patch_f = self.patch_FFT * kernel_f

            patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
            rec_patch = np.real(np.fft.ifft2(patch_f))


            plt.subplot(121), plt.imshow(self.patch, cmap='gray')
            plt.title('Input patch'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(rec_patch, cmap='gray')
            plt.title('filtered patch'), plt.xticks([]), plt.yticks([])
            plt.show()

    def reconstruction(self,weight=None):
        f_ifft = np.fft.ifftshift(self.patch_FFT)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(f_ifft))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch

    def gabor_filtering(self,theta,fre,weight=None):

        patch_size = self.patch_FFT.shape
        kernel = gabor_kernel(fre, theta=theta,sigma_x=4,sigma_y=4)

        f = kernel.real
        f = f - np.mean(f)
        f = f / (np.linalg.norm(f)+0.0001)


        kernel_f = np.fft.fft2(f,patch_size)
        kernel_f = np.fft.fftshift(kernel_f)
        patch_f = self.patch_FFT*kernel_f

        patch_f = np.fft.ifftshift(patch_f)  # *np.sqrt(np.abs(fshift)))
        rec_patch = np.real(np.fft.ifft2(patch_f))
        if weight is not None:
            rec_patch = rec_patch * weight
        return rec_patch


def show_orientation_field(img,dir_map,mask=None,fname=None):
    h,w = img.shape[:2]

    if mask is None:
        mask = np.ones((h,w),dtype=np.uint8)
    blkH, blkW = dir_map.shape

    blk_size = h/blkH

    R = blk_size/2*0.8
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    for i in range(blkH):
        y0 = i*blk_size + blk_size/2
        y0 = int(y0)
        for j in range(blkW):
            x0 = j*blk_size + blk_size/2
            x0 = int(x0)
            ori = dir_map[i,j]
            if mask[y0,x0] == 0:
                continue
            if ori<-9:
                continue
            x1 = x0 - R * math.cos(ori)
            x2 = x0 + R * math.cos(ori)
            y1 = y0 - R * math.sin(ori)
            y2 = y0 + R * math.sin(ori)
            plt.plot([x1, x2], [y1, y2], 'r-', lw=2)
    plt.axis('off')
    if fname is not None:
        fig.savefig(fname,dpi = 500, bbox_inches='tight', pad_inches = 0)
        plt.close()
    else:
        plt.show(block=True)
