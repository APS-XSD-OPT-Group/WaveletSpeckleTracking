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
import pywt
import os
import sys
import time
from PIL import Image
import glob
import scipy.constants as sc
from func import prColor, frankotchellappa, image_roi, Wavelet_transform, write_h5, write_json, find_disp
from euclidean_dist import dist_numba
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import multiprocessing as ms
import concurrent.futures
import copy


def background_remove(data, corner_size):
    '''
        use the four corner value as the background
    '''
    background_value = np.mean(data[0:corner_size, 0:corner_size] +
                               data[-corner_size - 1:-1, 0:corner_size] +
                               data[0:corner_size, -corner_size - 1:-1] +
                               data[-corner_size - 1:-1,
                                    -corner_size - 1:-1]) / 4
    return data - background_value


def load_images(Folder_path, filename_format='*.tif'):
    f_list = glob.glob(os.path.join(Folder_path, filename_format))
    f_list = sorted(f_list)
    img = []
    for f_single in f_list:
        img.append(np.array(Image.open(f_single)))
        # prColor('load image: {}'.format(f_single), 'green')
    if len(img) == 0:
        prColor('Error: wrong data path. No data is loaded.', 'red')
        sys.exit()

    return np.array(img)


def displace_wavelet(y_list,
                     img_wa_stack,
                     ref_wa_stack,
                     sub_pixel,
                     cal_half_window):
    '''
        calculate the coefficient of each pixel
    '''
    dim = img_wa_stack.shape
    disp_x = np.zeros((dim[0], dim[1]))
    disp_y = np.zeros((dim[0], dim[1]))

    # the axis for the peak position finding
    window_size = 2 * cal_half_window + 1
    y_axis = np.arange(window_size) - cal_half_window
    x_axis = np.arange(window_size) - cal_half_window
    XX, YY = np.meshgrid(x_axis, y_axis)

    for yy in range(dim[0]):
        for xx in range(dim[1]):
            img_wa_line = img_wa_stack[yy, xx, :]
            ref_wa_data = ref_wa_stack[yy:yy + window_size,
                                       xx:xx + window_size, :]

            Corr_img = dist_numba(img_wa_line, ref_wa_data)
 
            disp_y[yy, xx], disp_x[yy, xx] = find_disp(
                Corr_img, XX, YY, sub_resolution=True)

    return disp_y, disp_x, y_list


if __name__ == "__main__":
    if len(sys.argv) == 1:

        Folder_ref = 'H:/data/Jan2020_speckle/20200202/scan_speckle_exp_d500mm/sandpaper_5um/linear_rand10p_3um_Exp3s/refs/'
        Folder_img = 'H:/data/Jan2020_speckle/20200202/scan_speckle_exp_d500mm/sandpaper_5um/linear_rand10p_3um_Exp3s/sample_in/'
        Folder_result = 'H:/data/Jan2020_speckle/20200202/scan_speckle_exp_d500mm/sandpaper_5um/linear_rand10p_3um_Exp3s/wavelet_result_test/'
        # [image_size, cal_half_window, ues_parallel, n_group, n_cores, energy, pixel_size, distance, wavelet_cut_level]
        parameter_wavelet = [
            1500, 20, 1, 4, 4, 14e3, 0.65e-6, 500e-3, 2
        ]

    elif len(sys.argv) == 4:
        Folder_img = sys.argv[1]
        Folder_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        # [image_size, cal_half_window, ues_parallel, n_group, n_cores, energy, pixel_size, distance, wavelet_cut_level]
        parameter_wavelet = [
            1700, 40, 1, 4, 4, 14e3, 0.65e-6, 500e-3, 2
        ]
    elif len(sys.argv) == 14:
        Folder_img = sys.argv[1]
        Folder_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        parameter_wavelet = sys.argv[4:]
    else:
        prColor('Wrong parameters! should be: sample, ref, result', 'red')

    prColor('folder: {}'.format(Folder_result), 'green')
    # roi of the images
    M_image = int(parameter_wavelet[0])
    # do sub pixel calculation, if it's 1, no sub pixel
    pixel_sample = 1
    # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
    cal_half_window = int(parameter_wavelet[1])
    # parallel calculation or not
    use_parallel = int(parameter_wavelet[2])
    # process number for parallel
    n_cores = int(parameter_wavelet[3])
    # number to reduce the each memory use
    n_group = int(parameter_wavelet[4])

    # energy, 10kev
    energy = float(parameter_wavelet[5])
    wavelength = sc.value('inverse meter-electron volt relationship') / energy
    p_x = float(parameter_wavelet[6])
    z = float(parameter_wavelet[7])
    wavelet_level_cut = int(parameter_wavelet[8])

    ref_data = load_images(Folder_ref, '*.tif')
    img_data = load_images(Folder_img, '*.tif')

    # take out the roi
    ref_data = image_roi(ref_data, M_image)
    img_data = image_roi(img_data, M_image)

    if not os.path.exists(Folder_result):
        os.makedirs(Folder_result)
    sample_transmission = img_data[0] / ref_data[0]
    plt.imsave(os.path.join(Folder_result, 'transmission.png'),
               sample_transmission)

    # process the data to get the wavelet transform
    wavelet_method = 'db2'
    # wavelet wrapping level. 2 is half, 3 is 1/3 of the size
    max_wavelet_level = pywt.dwt_max_level(img_data.shape[0], wavelet_method)
    prColor('max wavelet level: {}'.format(max_wavelet_level), 'green')
    wavelet_level = max_wavelet_level
    coefs_level = wavelet_level + 1 - wavelet_level_cut

    if use_parallel:

        # multi-cores, parallel calculation
        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > n_cores:
            cores = n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        prColor('Process group number: {}'.format(n_group), 'light_purple')

        if cores * n_group > M_image:
            n_tasks = 4
        else:
            n_tasks = cores * n_group
        # split the y axis into small groups, all splitted in vertical direction
        y_axis = np.arange(img_data.shape[1])
        chunks_idx_y = np.array_split(y_axis, n_tasks)

        start_time = time.time()

        ref_data = ((ref_data - np.ndarray.mean(ref_data, axis=0)) /
                    np.ndarray.std(ref_data, axis=0))
        img_data = ((img_data - np.ndarray.mean(img_data, axis=0)) /
                    np.ndarray.std(img_data, axis=0))

        img_wa, level_name = Wavelet_transform(img_data,
                                           wavelet_method,
                                           w_level=wavelet_level,
                                           return_level=coefs_level)
        ref_wa, level_nam = Wavelet_transform(ref_data,
                                           wavelet_method,
                                           w_level=wavelet_level,
                                           return_level=coefs_level)

        prColor('vector length: {}\nUse wavelet coef: {}'.format(ref_wa.shape[2], level_name), 'green')
        
        del img_data
        del ref_data
        end_time = time.time()
        print('wavelet time: {}'.format(end_time - start_time))
        start_time = time.time()

        dim = img_wa.shape
        ref_wa_pad = np.pad(ref_wa,
                            ((cal_half_window, cal_half_window),
                             (cal_half_window, cal_half_window), (0, 0)),
                            'constant',
                            constant_values=(0, 0))

        # use CPU parallel to calculate
        result_list = []

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=cores) as executor:
            futures = []
            for y_list in chunks_idx_y:
                # get the stack data
                img_wa_stack = img_wa[y_list, :, :]
                ref_wa_stack = ref_wa_pad[y_list[0]:y_list[-1] +
                                          2 * cal_half_window + 1, :, :]

                # start the jobs
                futures.append(
                    executor.submit(displace_wavelet, y_list, img_wa_stack,
                                    ref_wa_stack, pixel_sample, 
                                    cal_half_window))

            for future in concurrent.futures.as_completed(futures):

                try:
                    result_list.append(future.result())
                    # display the status of the program
                    Total_iter = cores * n_group
                    Current_iter = len(result_list)
                    percent_iter = Current_iter / Total_iter * 100
                    str_bar = '>' * (int(np.ceil(
                        percent_iter / 2))) + ' ' * (int(
                            (100 - percent_iter) // 2))
                    prColor(
                        '\r' + str_bar + 'processing: [%3.1f%%] ' %
                        (percent_iter), 'purple')

                except:
                    prColor('Error in the parallel calculation', 'red')

        disp_y_list = [item[0] for item in result_list]
        disp_x_list = [item[1] for item in result_list]
        y_list = [item[2] for item in result_list]

        displace_y = np.zeros((dim[0], dim[1]))
        displace_x = np.zeros((dim[0], dim[1]))

        for y, disp_x, disp_y in zip(y_list, disp_x_list, disp_y_list):
            displace_x[y, :] = disp_x
            displace_y[y, :] = disp_y

        displace = [displace_y, displace_x]

    else:
        start_time = time.time()
        
        ref_data = ((ref_data - np.ndarray.mean(ref_data, axis=0)) /
                    np.ndarray.std(ref_data, axis=0))
        img_data = ((img_data - np.ndarray.mean(img_data, axis=0)) /
                    np.ndarray.std(img_data, axis=0))

        img_wa, level_name = Wavelet_transform(img_data,
                                           wavelet_method,
                                           w_level=wavelet_level,
                                           return_level=coefs_level)
        ref_wa, level_name = Wavelet_transform(ref_data,
                                           wavelet_method,
                                           w_level=wavelet_level,
                                           return_level=coefs_level)
        prColor('vector length: {}\nUse wavelet coef: {}'.format(ref_wa.shape[2], level_name), 'green')
        # delete the unused variables to save memory
        del img_data
        del ref_data
        end_time = time.time()
        print('wavelet time: {}'.format(end_time - start_time))
        start_time = time.time()

        dim = img_wa.shape
        XX, YY = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))

        ref_wa_pad = np.pad(ref_wa,
                            ((cal_half_window, cal_half_window),
                             (cal_half_window, cal_half_window), (0, 0)),
                            'constant',
                            constant_values=(1e10, 1e10))
        # get the stack data
        displace_y = np.zeros(YY.shape)
        displace_x = np.zeros(XX.shape)
        y_axis = np.arange(dim[0])
        if n_group > M_image:
            n_tasks = 4
        else:
            n_tasks = n_group
        chunks_idx_y = np.array_split(y_axis, n_tasks)
        for kk, y_list in enumerate(chunks_idx_y):
            # get the stack data
            img_wa_stack = img_wa[y_list, :, :]
            ref_wa_stack = ref_wa_pad[y_list[0]:y_list[-1] +
                                      2 * cal_half_window + 1, :, :]

            disp_y, disp_x, y_pos = displace_wavelet(
                y_list, img_wa_stack, ref_wa_stack,
                pixel_sample, cal_half_window)
            displace_y[y_pos, :] = disp_y
            displace_x[y_pos, :] = disp_x
            print(kk / len(chunks_idx_y))
            # print(displace_y[y,x], displace_x[y,x])
        displace = [displace_y, displace_x]

    end_time = time.time()
    prColor('\r' + 'Processing time: {:0.3f} s'.format(end_time - start_time),
            'light_purple')

    # remove the padding boundary of the displacement
    displace[0] = displace[0][cal_half_window:-cal_half_window,
                              cal_half_window:-cal_half_window]
    displace[1] = displace[1][cal_half_window:-cal_half_window,
                              cal_half_window:-cal_half_window]
    print('max of displace: {}, min of displace: {}'.format(
        np.amax(displace[0]), np.amin(displace[1])))

    DPC_y = (displace[0] - np.mean(displace[0])) * p_x / z
    DPC_x = (displace[1] - np.mean(displace[1])) * p_x / z

    phase = -frankotchellappa(DPC_x, DPC_y) * p_x * 2 * np.pi / wavelength

    if not os.path.exists(Folder_result):
        os.makedirs(Folder_result)

    plt.figure()
    plt.imshow(displace[0], cmap=cm.get_cmap('RdYlGn'), interpolation='bilinear', aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('[pixels]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'displace_x_colorbar.png'))

    plt.figure()
    plt.imshow(displace[1], cmap=cm.get_cmap('RdYlGn'), interpolation='bilinear', aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('[pixels]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'displace_y_colorbar.png'))

    plt.figure()
    plt.imshow(DPC_x*1e6, cmap=cm.get_cmap('RdYlGn'), interpolation='bilinear', aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('[$\mu rad$]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'dpc_x_colorbar.png'))

    plt.figure()
    plt.imshow(DPC_y*1e6, cmap=cm.get_cmap('RdYlGn'), interpolation='bilinear', aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('[$\mu rad$]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'dpc_y_colorbar.png'))

    plt.figure()
    plt.imshow(phase, cmap=cm.get_cmap('RdYlGn'), interpolation='bilinear', aspect='equal')
    cbar = plt.colorbar()
    cbar.set_label('[rad]', rotation=90)
    plt.savefig(os.path.join(Folder_result, 'phase_colorbar.png'))
    # plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(
        np.arange(phase.shape[1]) * p_x * 1e6,
        np.arange(phase.shape[0]) * p_x * 1e6)
    ax1.plot_surface(XX, YY, phase, cmap=cm.get_cmap('RdYlGn'))
    ax1.set_xlabel('x [$\mu m$]')
    ax1.set_ylabel('y [$\mu m$]')
    ax1.set_zlabel('phase [rad]')
    plt.savefig(os.path.join(Folder_result, 'Phase_3d.png'))

    # save the calculation results
    result_filename = 'WaveletSpeckle_' + str(M_image) + 'px_' + 'WaveletLevel_'+ str(wavelet_level) + '_CutLevel_' +str(wavelet_level_cut)
    write_h5(
        Folder_result, result_filename, {
            'displace_x': displace[1],
            'displace_y': displace[0],
            'DPC_x': DPC_x,
            'DPC_y': DPC_y,
            'phase': phase,
            'transmission_image': sample_transmission
        })

    parameter_dict = {
        'M_image': M_image,
        'sub_pixel': pixel_sample,
        'half_window': cal_half_window,
        'energy': energy,
        'wavelength': wavelength,
        'pixel_size': p_x,
        'z_distance': z,
        'parallel': use_parallel,
        'cpu_cores': cores,
        'n_group': n_group,
        'wavelet_method': wavelet_method,
        'wavelet_level': wavelet_level,
        'time_cost': end_time - start_time,
        'wavelet_level_cut': wavelet_level_cut
    }

    write_json(Folder_result, result_filename, parameter_dict)