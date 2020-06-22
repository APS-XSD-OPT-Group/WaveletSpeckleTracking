# #########################################################################
# Copyright (c) 2020, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2020. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

import numpy as np
import sys
import pywt
import os
import json
import h5py


def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if M == 0:
        return img
    elif len(img_size) == 2:
        if M > min(img_size):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[0] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1] + 1, pos_1[0]:pos_1[-1] + 1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[2] / 2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1] + 1,
                                  pos_1[0]:pos_1[-1] + 1]

    return img_data


def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)


def frankotchellappa(dpc_x, dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j * wx * fft2(dpc_x) - 1j * wy * fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator / denominator

    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi


# calculate the displacement from the correlation map
def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    '''
        Adapted from gaussiansubpixel.cpp by Zachary Taylor:
        https://github.com/OpenPIV/openpiv-c--qt/blob/master/src/gaussiansubpixel.cpp
        and a also a code by Ruxandra Cojocaru:
        https://gitlab.esrf.fr/cojocaru/swarp/-/blob/master/swarp/norm_xcorr.py

    '''

    # find the maximal value and postion
    Corr_max = np.amax(Corr_img)
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), Corr_img.shape)

    # Compute displacement on both axes
    Corr_img_pad = np.pad(Corr_img, ((1, 1), (1, 1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] -
          Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] +
           Corr_img_pad[max_pos_y - 1, max_pos_x] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] -
          Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] +
           Corr_img_pad[max_pos_y, max_pos_x - 1] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] -
           Corr_img_pad[max_pos_y + 1, max_pos_x - 1] -
           Corr_img_pad[max_pos_y - 1, max_pos_x + 1] +
           Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0

    if ((dxx * dyy - dxy * dxy) != 0.0):
        det = 1.0 / (dxx * dyy - dxy * dxy)
    else:
        det = 0.0
    # the XX, YY axis resolution
    pixel_res_x = XX_axis[0, 1] - XX_axis[0, 0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0, 0]
    Minor_disp_x = (-(dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (-(dxx * dy - dxy * dx) * det) * pixel_res_y

    if sub_resolution:
        disp_x = Minor_disp_x + XX_axis[pos[0], pos[1]]
        disp_y = Minor_disp_y + YY_axis[pos[0], pos[1]]
    else:
        disp_x = XX_axis[pos[0], pos[1]]
        disp_y = YY_axis[pos[0], pos[1]]

    max_x = XX_axis[0, -1]
    min_x = XX_axis[0, 0]
    max_y = YY_axis[-1, 0]
    min_y = YY_axis[0, 0]

    if disp_x > max_x:
        disp_x = max_y
    elif disp_x < min_x:
        disp_x = min_x

    if disp_y > max_y:
        disp_y = max_y
    elif disp_y < min_y:
        disp_y = min_y

    return disp_y, disp_x


def Wavelet_transform(img, wavelet_method='db2', w_level=1, return_level=1):
    '''
        do the wavelet transfrom for the 3D image data 
    '''
    coeffs = pywt.wavedec(img,
                          wavelet_method,
                          level=w_level,
                          mode='zero',
                          axis=0)

    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)

    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk + 1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]

    return coeffs_filter, level_name


def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name + '.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name,
                             data=data_dict[key_name],
                             compression="gzip",
                             compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name + '.hdf5'), 'green')


def read_h5(file_path, key_name, print_key=False):
    '''
        read the data with the key_name in the h5 file
    '''

    if not os.path.exists(file_path):
        prColor('Wrong file path', 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')

        data = f[key_name][:]
    return data


def write_json(result_path, file_name, data_dict):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name + '.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=0)

    prColor('result json file : {} saved'.format(file_name + '.json'), 'green')


def read_json(filepath, print_para=False):

    if not os.path.exists(filepath):
        prColor('Wrong file path', 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')

    return data