import os
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from AMC import Device
from Swabian import TimeTagger


# ---- Stage abstraction ----
class AMC_Axis:
    def __init__(self, dev, axis_id):
        self.dev = dev
        self.axis_id = axis_id

    def enable(self):
        self.dev.control.setControlOutput(self.axis_id, True)

    def disable(self):
        self.dev.control.setControlOutput(self.axis_id, False)

    def move_abs(self, pos_um):
        pos_nm = int(pos_um * 1e3)
        self.dev.move.setControlTargetPosition(self.axis_id, pos_nm)
        self.dev.control.setControlMove(self.axis_id, True)
        while not self.dev.status.getStatusTargetRange(self.axis_id):
            time.sleep(0.05)
        self.dev.control.setControlMove(self.axis_id, False)

    def get_pos(self):
        return self.dev.move.getPosition(self.axis_id) / 1e3


# ---- Scan routine ----
def sweep_multi(devices, starts, stops, npts, measure_func,
                 reset=True, wait_time=0.1, serpentine=False):
    """
    Sweeps devices[0] (X) and devices[1] (Y) over a rectangular grid and calls
    measure_func() at each point. No live plotting. Prints "scan started" /
    "scan Finished", plus a progress line every 50 points completed.

    Returns:
        results: (Npts, 4) array of [X, Y, Serial, Data]
        xlist, ylist: the X and Y coordinate arrays
    """
    xlist = np.linspace(starts[0], stops[0], npts[0])
    ylist = np.linspace(starts[1], stops[1], npts[1])

    Npts = npts[0] * npts[1]
    results = np.zeros((Npts, 4))

    # build scan order (with optional serpentine)
    scan_points = []
    for j, y in enumerate(ylist):
        row = [(x, y) for x in xlist]
        if serpentine and (j % 2 == 1):
            row.reverse()
        scan_points.extend(row)

    print("scan started")

    for idx, (x, y) in enumerate(scan_points):
        devices[0].move_abs(x)
        devices[1].move_abs(y)
        time.sleep(wait_time)

        val = measure_func()
        results[idx] = (x, y, idx + 1, val)

        if (idx + 1) % 50 == 0:
            print(f"{idx + 1}/{Npts} points done")

    print("scan Finished")

    if reset:
        devices[0].move_abs(starts[0])
        devices[1].move_abs(starts[1])

    return results, xlist, ylist


# ---- Plotting helper ----
def plot_apd_data(data_array, xlist, ylist, title='APD Counts', cmap='viridis'):
    """
    Renders the final map. Expects data_array[:, 3] to be the measured value
    in row-major (serial) order. Returns (fig, ax, im) so a click-to-move
    handler can be attached afterwards.
    """
    nX, nY = len(xlist), len(ylist)
    z = data_array[:, 3].reshape(nY, nX)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        z,
        extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
        origin='lower',
        aspect='auto',
        cmap=cmap
    )
    fig.colorbar(im, ax=ax, label='Counts')
    ax.set_xlabel('X (\u00b5m)')
    ax.set_ylabel('Y (\u00b5m)')
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax, im


def enable_click_to_move(fig, ax, x_stage, y_stage):
    """
    Connects a click handler to `ax`: clicking anywhere on the map moves the
    stage to that (X, Y) position in µm. Clicks made while the toolbar is in
    zoom/pan mode, or outside the axes, are ignored.
    """
    def on_click(event):
        if event.inaxes != ax:
            return
        toolbar = fig.canvas.toolbar
        if toolbar is not None and toolbar.mode != '':
            return  # zooming/panning, not a move request
        if event.xdata is None or event.ydata is None:
            return
        x_stage.move_abs(event.xdata)
        y_stage.move_abs(event.ydata)

    return fig.canvas.mpl_connect('button_press_event', on_click)


# ---- Main scan routine ----
def run_full_scan(click_to_move=True):
    # connect to AMC100
    amc = Device("192.168.1.1")    # <- replace with your controller's IP
    amc.connect()
    x_stage = AMC_Axis(amc, axis_id=1)
    y_stage = AMC_Axis(amc, axis_id=2)

    x_stage.enable()
    y_stage.enable()

    # set up TimeTagger
    tagger = TimeTagger.createTimeTagger()
    tagger.setTriggerLevel(channel=1, voltage=1.0)
    tagger.setTriggerLevel(channel=2, voltage=1.0)
    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[1, 2],
        binwidth=int(2e10),
        n_values=10
    )

    def get_APD_counts():
        duration_s = 0.21
        duration_ps = int(duration_s * 1e12)
        counter.startFor(capture_duration=duration_ps)
        counter.waitUntilFinished()
        data = counter.getData()
        return int(data[0].sum())

    # scan parameters (unchanged from original)
    xdomain = [-5, 5]
    ydomain = [-5, 5]
    pts = [25, 25]

    data_array, xlist, ylist = sweep_multi(
        devices=[x_stage, y_stage],
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_APD_counts,
        wait_time=0.1,
        serpentine=True,
    )

    # save results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder = r"C:\Student Folders\Iftekhar\PL Maps"
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, f"{timestamp}_PL_Map_data.txt")

    header = "X(um)\tY(um)\tSerial\tData\n"
    np.savetxt(
        file_name,
        data_array,
        fmt=["%0.3f", "%0.3f", "%d", "%d"],
        delimiter="\t",
        header=header,
        comments=""
    )

    # final plot (rendered once, after the scan is complete)
    fig, ax, im = plot_apd_data(data_array, xlist, ylist, title="APD Fluorescence Map")

    if click_to_move:
        enable_click_to_move(fig, ax, x_stage, y_stage)

    plt.show()

    return file_name, data_array


if __name__ == "__main__":
    run_full_scan()