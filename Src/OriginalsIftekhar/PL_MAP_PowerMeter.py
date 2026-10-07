import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from AMC import Device            # or: from attocube.amc import AMCDevice
import pyvisa

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
        fig, ax = plt.subplots(figsize=(7, 6))
        Z = np.full((npts[1], npts[0]), np.nan)
        im = ax.imshow(
            Z,
            extent=[xlist[0], xlist[-1], ylist[0], ylist[-1]],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power (W)')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('Real-Time Optical Power Map')
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
            print(f"[{idx + 1}/{Npts}] x={x:.3f} y={y:.3f} | Time={pt_time:.2f}s → {val:.4e} W")

    total_time = time.perf_counter() - t0
    print(f"Scan complete: {Npts} pts in {total_time:.1f}s")

    # return to start
    if reset:
        devices[0].move_abs(starts[0])
        devices[1].move_abs(starts[1])

    # finalize the plot
    if live_plot:
        ax.set_title('Final Optical Power Map')
        plt.ioff()
        plt.show()

    return results, xlist, ylist


# ---- Main scan routine ----
def run_full_scan():
    # connect to AMC100
    amc = Device("192.168.1.1")    # ← replace with your controller's IP
    amc.connect()
    x_stage = AMC_Axis(amc, axis_id=1)
    y_stage = AMC_Axis(amc, axis_id=2)
    # z_stage = AMC_Axis(amc, axis_id=3)  # assuming axis 3 controls Z;;;;;;; reposition

    x_stage.enable()
    y_stage.enable()
    # z_stage.enable()

    # ---- set up Thorlabs PM100D power meter (SCPI over VISA) ----
    rm = pyvisa.ResourceManager()
    print("Available resources:")
    print(rm.list_resources())

    # !!! Update the USB resource string (serial number / PID) to match your connected device.
    power_meter = rm.open_resource('USB0::0x1313::0x8078::P0042558::0::INSTR')
    print(power_meter.query("SYST:SENS:IDN?"))

    power_meter.write("SENS:RANGE:AUTO ON")     # auto-ranging
    power_meter.write("SENS:CORR:WAV 532")      # set wavelength for correct calibration
    power_meter.write("SENS:POW:UNIT W")        # units in Watts
    power_meter.write("SENS:AVER:200")          # averaging

    def get_power_reading():
        return float(power_meter.query("MEAS:POW?"))

    # scan parameters
    xdomain = [-1200, 1200]
    ydomain = [-1200, 1200]
    pts = [48, 48]

    # run the sweep
    data_array, xlist, ylist = sweep_multi(
        devices=[x_stage, y_stage],
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_power_reading,
        wait_time=0.2,
        serpentine=True,
        verbose=True,
        live_plot=True
    )

    # disable axes
    # x_stage.disable()
    # y_stage.disable()

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    text_file_name = f"{timestamp}_optical_power_meter_scan.txt"
    folder = r"C:\\Student Folders\\Iftekhar\\beam diameter"
    os.makedirs(folder, exist_ok=True)

    file_name = os.path.join(folder, text_file_name)

    # save to text file (4 columns: X, Y, Serial, Power)
    header = "X(um)\tY(um)\tSerial\tPower(W)\n"
    np.savetxt(
        file_name,
        data_array,
        fmt=["%0.3f", "%0.3f", "%d", "%.6e"],
        delimiter="\t",
        header=header,
        comments=""
    )

    print(f"Results saved to {file_name}")

    # close power meter / VISA session
    power_meter.close()
    rm.close()

    return file_name, data_array


if __name__ == "__main__":
    run_full_scan()