import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import  datetime
from AMC import Device            # or: from attocube.amc import AMCDevice
from Swabian import TimeTagger                 # pip‑installable: python‑timetagger
from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

sdk = atmcd()  # Load the atmcd library
codes = atmcd_codes

ret = sdk.Initialize("")  # Initialize camera
print("Function Initialize returned {}".format(ret))

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
        #while not self.dev.status.getStatusTargetRange(self.axis_id):
            #time.sleep(0.05)
        #self.dev.control.setControlMove(self.axis_id, False)

    def get_pos(self):
        return self.dev.move.getPosition(self.axis_id) / 1e3



#///// realtime

def sweep_multi(devices, starts, stops, npts, measure_func,
                reset=True, wait_time=0.1,
                serpentine=False, verbose=True,
                live_plot=False):
    """
    If live_plot=True, opens an interactive figure and updates it on each point.
    Uses the default 'viridis' colormap and auto color-scaling.
    """
    # build coords
    xlist = np.linspace(starts[0], stops[0], npts[0])
    ylist = np.linspace(starts[1], stops[1], npts[1])

    # pre-allocate results array: [x, y, serial, data]
    Npts = npts[0] * npts[1]
    results = np.zeros((Npts, 4))

    # build scan order (with optional serpentine)
    scan_points = []
    for j, y in enumerate(ylist):
        row = [(x, y) for x in xlist]
        if serpentine and (j % 2 == 1):
            row.reverse()
        scan_points.extend(row)

    # set up live plot
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7,6))
        Z = np.full((npts[1], npts[0]), np.nan)
        im = ax.imshow(
            Z,
            extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Counts')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('Real‑Time PL Map')
        plt.show()

    t0 = time.perf_counter()

    for idx, (x, y) in enumerate(scan_points):
        pt_start = time.perf_counter()
        # Move X and Y stages
        devices[0].move_abs(x)
        devices[1].move_abs(y)
        time.sleep(wait_time)

        val = measure_func()

        pt_time = time.perf_counter() - pt_start
        results[idx] = (x, y, idx + 1, val)

        # Update live plot and verbose print as before
        if live_plot:
            ix = int(round((x - xlist[0]) / (xlist[-1] - xlist[0]) * (npts[0] - 1)))
            iy = int(round((y - ylist[0]) / (ylist[-1] - ylist[0]) * (npts[1] - 1)))
            Z[iy, ix] = val
            im.set_data(Z)
            im.autoscale()
            fig.canvas.draw()
            fig.canvas.flush_events()

        if verbose:
            print(f"[{idx + 1}/{Npts}] x={x:.3f} y={y:.3f} | Time={pt_time:.2f}s → {val}")

    total_time = time.perf_counter() - t0
    print(f"Scan complete: {Npts} pts in {total_time:.1f}s")

    # return to start
    if reset:
        devices[0].move_abs(starts[0])
        devices[1].move_abs(starts[1])

    # finalize the plot
    if live_plot:
        ax.set_title('Final PL Map')
        plt.ioff()
        plt.show()

    return results, xlist, ylist


# ---- Main scan routine ----
def run_full_scan():
    # --- collect scan metadata from user ---
    laser_power_uw = float(input("Laser power (uW): "))
    focus = input("Focus level: ");
    filter_used = input("Filter used: ")
    count_before_filter = input("Count before filter: ")
    count_after_filter = input("Count after filter: ")
    nw_diameter = input("nw diameter: ")
    # connect to AMC100
    amc = Device("192.168.1.1")    # ← replace with your controller’s IP
    amc.connect()
    x_stage = AMC_Axis(amc, axis_id=1)
    y_stage = AMC_Axis(amc, axis_id=2)
    # z_stage = AMC_Axis(amc, axis_id=3)  # assuming axis 3 controls Z;;;;;;; reposition

    x_stage.enable()
    y_stage.enable()
    # z_stage.enable()

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
        duration_s = 0.2
        duration_ps = int(duration_s * 1e12)
        counter.startFor(capture_duration=duration_ps)
        counter.waitUntilFinished()
        data = counter.getData()
        return int(data[0].sum())


    def plot_apd_data(data_array, xlist, ylist, ch_index=0, title='APD Counts'):
        """
        Expects data_array[:, 3] to be the measured value.
        ch_index isn’t used here since we collapsed to a single channel.
        """
        nX, nY = len(xlist), len(ylist)
        # reshape by serial order: row-major
        z = data_array[:, 3].reshape(nY, nX)
        plt.figure(figsize=(8, 6))
        plt.imshow(
            z,
            extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label='Counts')
        plt.xlabel('X (µm)')
        plt.ylabel('Y (µm)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(
            os.path.join(folder, text_file_name.replace(".txt", ".png")),
            dpi=600,
            format="png"
        )
        plt.show()

    # scan parameters
    xdomain = [-4,4]
    ydomain = [-4,4]
    pts = [32,32]

    # run the sweep
    data_array, xlist, ylist = sweep_multi(
        devices=[x_stage, y_stage],
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_APD_counts,
        wait_time=0.1,
        serpentine=True,
        verbose=True,
        live_plot=True
    )

    # disable axes
    # x_stage.disable()
    # y_stage.disable()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    text_file_name = f"{timestamp}_{nw_diameter}_{laser_power_uw}_{focus}_{filter_used}_{count_before_filter}_{count_after_filter}_PL_Map_data.txt"
    folder = r"C:\\Student Folders\\Iftekhar\\Aug 2026\\After Gabriel Alignment"
    os.makedirs(folder, exist_ok=True)

    file_name = os.path.join(folder, text_file_name)

    # save to text file (4 columns: X, Y, Serial, Data)
    header = "X(um)\tY(um)\tSerial\tData\n"
    np.savetxt(
        file_name,
        data_array,
        fmt=["%0.3f", "%0.3f", "%d", "%d"],
        delimiter="\t",
        header=header,
        comments=""
    )

    # optional: plot the map
    # plot_apd_data(data_array, xlist, ylist, title="APD Fluorescence Map")

    print(f"Results saved to {file_name}")
    return file_name, data_array


if __name__ == "__main__":
    run_full_scan()