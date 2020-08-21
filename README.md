# Multi-resolution wavelet speckle tracking
## Multi-resolution WSVT
Multi resolution WSVT (Wavelet-transform-based speckle vector tracking) is a package to solve speckle vector tracking problem, where the discrete wavelet transform and multi-resolution analysis are implemented to accelerate the process and increase noise robustness.  
## Multi-resolution WXST
Multi resolution WXST (Wavelet-transform-based single shot speckle tracking) is a package to solve single shot speckle tracking problem, where the discrete wavelet transform and multi-resolution analysis are implemented to accelerate the process and increase noise robustness.  

To install:\
            `python setup.py install`
  
Package required:\
            `numpy`, `scipy`, `pywt`, `h5py`, `numba`, `PIL`, `json`

To run the script:\
            `python speckle_wavelet.py path_sample path_ref path_result parameters`

parameters:\
            `image_size: image size`\
            `cal_half_window: searching window size`\
            `N_s_extend: searching widnow size for pyramid levels except the maximum pyramid level. It can be 2 or 4 generally` \
            `template_window: template window size to find the displacement`\
            `ues_parallel: CPU parallel, True or False`\
            `n_group: split data into n_group`\
            `n_cores: cpu cores used`\
            `energy: X-ray energy [eV]`\
            `pixel_size: detector pixel size [m]`\
            `distance: speckle to detector distance [m]`\
            `wavelet_cut_level: cut the wavelet coefficients, 0 is no cut`\
            `pyramid level: multi-resollution analysis level, such as 2 or 3`\
            `n_iter: iteration for speckle tracking, normally not needed, setting to 1`\
            `use_wavelet: use wavelet transform or not`

