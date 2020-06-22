# WSVT

WSVT (Wavelet-transform-based speckle vector tracking) is a package to solve speckle vector tracking problem, where the discrete wavelet transform is implemented to accelerate the process and increase noise robustness.  

To install:\
            `python setup.py install`
  
Package required:\
            `numpy`, `scipy`, `pywt`, `h5py`, `numba`, `PIL`, `json`

To run the script:\
            `python speckle_wavelet.py path_sample path_ref path_result parameters`

parameters:\
            `image_size: image size`\
            `cal_half_window: searching window`\
            `ues_parallel: CPU parallel, True or False`\
            `n_group: split data into n_group`\
            `n_cores: cpu cores used`\
            `energy: X-ray energy [eV]`\ `pixel_size: detector pixel size [m]`\
            `distance: speckle to detector distance [m]`\
            `wavelet_cut_level: cut the wavelet coefficients, 0 is no cut`
