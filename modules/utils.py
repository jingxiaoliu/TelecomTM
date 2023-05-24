import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.stats import norm
from obspy.signal.filter import bandpass


def _cut_taper(data, t_axis):
    nt = data.shape[-1]
    taper_len = np.argmin(np.abs(t_axis))
    t_axis = t_axis[taper_len: nt - taper_len]
    data = data[:, taper_len: nt - taper_len]
    return data, t_axis

def _read_das_npz(fname, **kwargs):
    # read npz file and return das data and its spatil and time axis
    try:
        data_file = np.load(fname)
    except:
        raise Exception(f"fname: {fname}")
    data = data_file["data"]
    x_axis = data_file["x_axis"]

    t_axis = data_file["t_axis"]
    ch1 = kwargs.get('ch1', x_axis[0])
    ch2 = kwargs.get('ch2', x_axis[-1])

    ch1_idx = np.argmax(x_axis >= ch1)
    ch2_idx = np.argmax(x_axis >= ch2)
    data = data[ch1_idx:ch2_idx]

    cut_taper = kwargs.get("cut_taper", True)
    if cut_taper:
        data, t_axis = _cut_taper(data, t_axis)
    return data, x_axis[ch1_idx:ch2_idx], t_axis

def das_preprocess(data_in):
    # Signal detrend and median normalization
    data_out = signal.detrend(data_in)
    data_out = data_out - np.median(data_out, axis=0)
    return data_out

def plot_data(data, x_axis, t_axis, pclip=98, ax=None, figsize=(10, 10), 
                y_lim=None, x_lim=None, fig_name=None, 
                fig_dir="Fig/", fontsize=16, tickfont=12):
    # Plot das data and save the figure
    vmax = np.percentile(np.abs(data), pclip)
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(data.T,
              aspect="auto",
              extent=[x_axis[0], x_axis[-1], t_axis[-1], t_axis[0]],
              cmap="gray",
              vmax=vmax,
              vmin=-vmax)
    ax.plot()
    ax.set_ylim(y_lim)
    ax.set_xlim(x_lim)
    ax.set_xlabel("1 m per channel", fontsize=fontsize)
    ax.set_ylabel("Time [s]", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tickfont)
    if fig_name:
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path)
