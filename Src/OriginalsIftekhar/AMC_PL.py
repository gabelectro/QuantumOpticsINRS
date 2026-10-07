import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import  datetime
from AMC import Device            # or: from attocube.amc import AMCDevice
from Swabian import TimeTagger                 # pip‑installable: python‑timetagger

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
        if idx == 0:
            ny = y
            devices[1].move_abs(y)
        else:
            if ny!=y:
                devices[1].move_abs(y)
                ny = y
        time.sleep(wait_time)

        val = measure_func()
        #
        # retry_count = 0
        # while val < count_threshold and retry_count < max_retries:
        #     print(f"Count {val} below threshold {count_threshold}, repositioning Z...")
        #     # Move Z by a small step upward (e.g., 0.1 µm); adjust displacement as needed
        #     current_z = devices[2].get_pos()
        #     devices[2].move_abs(current_z + 0.2)
        #     time.sleep(wait_time)
        #     val = measure_func()
        #     retry_count += 1
        #
        # if val < count_threshold:
        #     print(f"Warning: Failed to recover count after {max_retries} retries at point {idx + 1}. Continuing scan.")

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

# ---- Plotting helper ----

# ---- Main scan routine ----
def run_full_scan():
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
        duration_s = 0.25
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
        plt.show()

    # scan parameters
    # xdomain = [, 4]
    # ydomain = [-4, 4]
    # pts     = [20,20]

    # scan parameters
    xdomain = [-6, 6]
    ydomain = [-6,6]
    pts = [30, 30]
    # ensure output folder

    # run the sweep
    data_array, xlist, ylist = sweep_multi(
        devices=[x_stage, y_stage], #repos
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_APD_counts,
        wait_time=0.1,
        serpentine=True,
        verbose=True,
        live_plot=True  # ← just turn on live plotting
    )

    # disable axes
    # x_stage.disable()
    # y_stage.disable()
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    text_file_name = f"{timestamp}_PL_Map_data.txt";
    folder = r"C:\\Student Folders\\Iftekhar\\PL Maps"
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
        plt.show()

    # optional: plot the map
    plot_apd_data(data_array, xlist, ylist, title="APD Fluorescence Map")

    print(f"Results saved to {file_name}")
    return file_name, data_array


if __name__ == "__main__":
    run_full_scan()


#/....................Optimize code 15 aug/////////////////////////////////////////////////////////////////////////
# count threshold more logically, textfile output containz z info
#
# import numpy as np
# import time
# import os
# import matplotlib.pyplot as plt
#
# from AMC import Device            # or: from attocube.amc import AMCDevice
# import TimeTagger                 # pip‑installable: python‑timetagger
#
# # ---- Stage abstraction ----
#
# class AMC_Axis:
#     def __init__(self, dev, axis_id):
#         self.dev = dev
#         self.axis_id = axis_id
#
#     def enable(self):
#         self.dev.control.setControlOutput(self.axis_id, True)
#
#     def disable(self):
#         self.dev.control.setControlOutput(self.axis_id, False)
#
#     def move_abs(self, pos_um):
#         pos_nm = int(pos_um * 1e3)
#         self.dev.move.setControlTargetPosition(self.axis_id, pos_nm)
#         self.dev.control.setControlMove(self.axis_id, True)
#         while not self.dev.status.getStatusTargetRange(self.axis_id):
#             time.sleep(0.05)
#         self.dev.control.setControlMove(self.axis_id, False)
#
#     def get_pos(self):
#         return self.dev.move.getPosition(self.axis_id) / 1e3
#
#
# # ///// Realtime
#
# def sweep_multi(devices, starts, stops, npts, measure_func,
#                 reset=True, wait_time=0.1,
#                 serpentine=False, verbose=True,
#                 live_plot=False, count_threshold=None):
#     """
#     If live_plot=True, opens an interactive figure and updates it on each point.
#     Uses the default 'viridis' colormap and auto color-scaling.
#     """
#     # build coords
#     xlist = np.linspace(starts[0], stops[0], npts[0])
#     ylist = np.linspace(starts[1], stops[1], npts[1])
#
#     # pre-allocate results array: [x, y, serial, data, z]
#     Npts = npts[0] * npts[1]
#     results = np.zeros((Npts, 5))
#
#     # build scan order (with optional serpentine)
#     scan_points = []
#     for j, y in enumerate(ylist):
#         row = [(x, y) for x in xlist]
#         if serpentine and (j % 2 == 1):
#             row.reverse()
#         scan_points.extend(row)
#
#     # set up live plot
#     if live_plot:
#         plt.ion()
#         fig, ax = plt.subplots(figsize=(7,6))
#         Z = np.full((npts[1], npts[0]), np.nan)
#         im = ax.imshow(
#             Z,
#             extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
#             origin='lower',
#             aspect='auto',
#             cmap='viridis'
#         )
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label('Counts')
#         ax.set_xlabel('X (µm)')
#         ax.set_ylabel('Y (µm)')
#         ax.set_title('Real‑Time PL Map')
#         plt.show()
#
#     t0 = time.perf_counter()
#
#     # Initial Z position assumption
#     initial_z = 0.0
#
#     # Scan the first two rows to determine the threshold
#     row_counts = []
#     for idx, (x, y) in enumerate(scan_points[:npts[0]*2]):  # First two rows
#         devices[0].move_abs(x)
#         devices[1].move_abs(y)
#         time.sleep(wait_time)
#         val = measure_func()
#         row_counts.append(val)
#
#     # Calculate the threshold based on the first two rows' counts (mean of first two rows)
#     mean_counts = np.mean(row_counts)
#     std_counts = np.std(row_counts)
#     count_threshold = mean_counts - std_counts  # Set threshold below the mean
#
#     print(f"Calculated count threshold: {count_threshold}")
#     max_try = 0
#     for idx, (x, y) in enumerate(scan_points):
#         pt_start = time.perf_counter()
#         # Move X and Y stages
#         devices[0].move_abs(x)
#         devices[1].move_abs(y)
#         time.sleep(wait_time)
#
#         val = measure_func()
#         z_pos = initial_z  # Start with initial Z position (assumed to be 0.0)
#
#         retry_count = 0
#
#         while val < count_threshold and retry_count < 6:  # Max retries
#             print(f"Count {val} below threshold {count_threshold}, repositioning Z...")
#             # Move Z by a small step upward (e.g., 0.2 µm); adjust displacement as needed
#             z_pos += 0.2
#             devices[2].move_abs(z_pos)
#             time.sleep(wait_time)
#             val = measure_func()
#             retry_count += 1
#         # initial_z=z_pos  #---------recurring
#         if val < count_threshold:
#             print(f"Warning: Failed to recover count after 5 retries at point {idx + 1}. Continuing scan.")
#             max_try += 1
#             if max_try > 7:
#                 initial_z += 0.5 # brute shifts upward
#                 print(f"!!!!3 warning, increasing Z {idx + 1}")
#                 max_try = 0
#
#
#
#         pt_time = time.perf_counter() - pt_start
#         results[idx] = (x, y, idx + 1, val, z_pos)
#
#         # Update live plot and verbose print as before
#         if live_plot:
#             ix = int(round((x - xlist[0]) / (xlist[-1] - xlist[0]) * (npts[0] - 1)))
#             iy = int(round((y - ylist[0]) / (ylist[-1] - ylist[0]) * (npts[1] - 1)))
#             Z[iy, ix] = val
#             im.set_data(Z)
#             im.autoscale()
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#
#         if verbose:
#             print(f"[{idx + 1}/{Npts}] x={x:.3f} y={y:.3f} | Time={pt_time:.2f}s → {val}")
#
#     total_time = time.perf_counter() - t0
#     print(f"Scan complete: {Npts} pts in {total_time:.1f}s")
#
#     # return to start
#     if reset:
#         devices[0].move_abs(starts[0])
#         devices[1].move_abs(starts[1])
#
#     # finalize the plot
#     if live_plot:
#         ax.set_title('Final PL Map')
#         plt.ioff()
#         plt.show()
#
#     return results, xlist, ylist
#
# # ---- Plotting helper ----
#
# # ---- Main scan routine ----
# def run_full_scan():
#     # connect to AMC100
#     amc = Device("192.168.1.1")    # ← replace with your controller’s IP
#     amc.connect()
#     # # //// axis check
#     # for axis_id in range(4):  # Check up to axis 3
#     #     try:
#     #         amc.control.setControlOutput(axis_id, True)
#     #         print(f"Axis {axis_id} is valid and enabled.")
#     #         amc.control.setControlOutput(axis_id, False)
#     #     except Exception as e:
#     #         print(f"Axis {axis_id} is invalid: {e}")
#     # #///////
#     x_stage = AMC_Axis(amc, axis_id=1)
#     y_stage = AMC_Axis(amc, axis_id=2)
#     z_stage = AMC_Axis(amc, axis_id=0)  # assuming axis 3 controls Z
#
#     x_stage.enable()
#     y_stage.enable()
#     z_stage.enable()
#
#     # set up TimeTagger
#     tagger = TimeTagger.createTimeTagger()
#     tagger.setTriggerLevel(channel=1, voltage=1.0)
#     tagger.setTriggerLevel(channel=2, voltage=1.0)
#     counter = TimeTagger.Counter(
#         tagger=tagger,
#         channels=[1, 2],
#         binwidth=int(1.5e10),
#         n_values=10
#     )
#
#
#     def get_APD_counts():
#         duration_s = 0.152
#         duration_ps = int(duration_s * 1e12)
#         counter.startFor(capture_duration=duration_ps)
#         counter.waitUntilFinished()
#         data = counter.getData()
#         return int(data[0].sum())
#
#     # scan parameters
#     xdomain = [-8,8]
#     ydomain = [-8,8]
#     pts     = [40,40]
#
#     # run the sweep
#     data_array, xlist, ylist = sweep_multi(
#         devices=[x_stage, y_stage, z_stage],
#         starts=[xdomain[0], ydomain[0]],
#         stops=[xdomain[1], ydomain[1]],
#         npts=pts,
#         measure_func=get_APD_counts,
#         wait_time=0.1,
#         serpentine=True,
#         verbose=True,
#         live_plot=True
#     )
#
#     folder = "Fardin_27Aug_MBE"
#     os.makedirs(folder, exist_ok=True)
#     file_name = os.path.join(folder, "INminus_0p75mW_power_CWL727_withopt.txt")
#
#     # save to text file (5 columns: X, Y, Serial, Data, Z)
#     header = "X(um)\tY(um)\tSerial\tData\tZ(um)\n"
#     np.savetxt(
#         file_name,
#         data_array,
#         fmt=["%0.3f", "%0.3f", "%d", "%d", "%0.3f"],
#         delimiter="\t",
#         header=header,
#         comments=""
#     )
#
#     def plot_apd_data(data_array, xlist, ylist, ch_index=0, title='APD Counts'):
#         """
#                     Expects data_array[:, 3] to be the measured value.
#                     ch_index isn’t used here since we collapsed to a single channel.
#                     """
#         nX, nY = len(xlist), len(ylist)
#         # reshape by serial order: row-major
#         z = data_array[:, 3].reshape(nY, nX)
#         plt.figure(figsize=(8, 6))
#         plt.imshow(
#             z,
#             extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
#             origin='lower',
#             aspect='auto',
#             cmap='viridis'
#         )
#         plt.colorbar(label='Counts')
#         plt.xlabel('X (µm)')
#         plt.ylabel('Y (µm)')
#         plt.title(title)
#         plt.tight_layout()
#         plt.show()
#
#     # optional: plot the map
#     plot_apd_data(data_array, xlist, ylist, title="APD Fluorescence Map")
#
#     print(f"Results saved to {file_name}")
#     return file_name, data_array
#
#
# if __name__ == "__main__":
#     run_full_scan()
