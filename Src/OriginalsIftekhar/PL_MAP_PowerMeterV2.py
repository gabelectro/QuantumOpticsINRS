import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
from datetime import datetime
from AMC import Device            # or: from attocube.amc import AMCDevice
import pyvisa

# ---- Stage abstraction ----
class AMC_Axis:
    def __init__(self, dev, axis_id):
        self.dev = dev
        self.axis_id = axis_id
        self.ref_pos = self.dev.control.getReferencePosition(self.axis_id)
        self.amplitude = self.dev.control.getControlAmplitude(self.axis_id)  # Get the amplitude of the axis
        self.range = self.dev.control.getControlTargetRange(self.axis_id)  # Get the range of the axis
        self.freq = self.dev.control.getControlFrequency(self.axis_id)  # Get the frequency of the axis
        self.steptime = 1.0 / self.freq  # Calculate the step time based on frequency
    def enable(self):
        self.dev.control.setControlOutput(self.axis_id, True)

    def disable(self):
        self.dev.control.setControlOutput(self.axis_id, False)

    def closedloop(self,enable=True):
        self.dev.control.setControlMove(self.axis_id, enable)

    def move_abs(self, target_um,fastscan = True,maxerr_um = 0.1,stbcheck = False):
        initime = time.time()
        pos_nm = int(target_um * 1e3)
        perror = 0
        basetimeout = 3.0  # seconds
        moving = False
        self.dev.move.setControlTargetPosition(self.axis_id, pos_nm)
        pos_um = self.dev.move.getPosition(self.axis_id) / 1e3
        perror = pos_um - target_um
        estdur = perror * self.steptime #1 step = 1 um
        timeout = estdur + basetimeout  # seconds
        #while not self.dev.status.getStatusTargetRange(self.axis_id):
        while self.dev.status.getStatusMoving(self.axis_id):
            moving = True
            if abs(perror) < maxerr_um:
                if fastscan:
                    break
                pos_um = self.dev.move.getPosition(self.axis_id) / 1e3
                perror = pos_um - target_um 
            if time.time() - initime > timeout: 
                print(f"Timeout: Axis {self.axis_id} did not reach target position within {timeout} seconds.")
                break
        if not fastscan:
            while abs(perror) < maxerr_um:
                pos_um = self.dev.move.getPosition(self.axis_id) / 1e3
                perror = pos_um - target_um 
                if time.time() - initime > timeout: 
                    print(f"Timeout: Axis {self.axis_id} did not reach target position within {timeout} seconds.")
                    break
        return perror,moving

    def get_pos(self):
        return self.dev.move.getPosition(self.axis_id) / 1e3


#///// realtime

def sweep_multi(devices, starts, stops, npts, measure_func,filename,
                reset=True, wait_time=0.05,
                serpentine=False, verbose=True,
                live_plot=False):
    """
    If live_plot=True, opens an interactive figure and updates it on each point.
    Uses the default 'viridis' colormap and auto color-scaling.
    """
    # build coords
    xlist = np.linspace(starts[0], stops[0], npts[0])
    ylist = np.linspace(starts[1], stops[1], npts[1])

    # pre-allocate results array: [target x, act x, target y, act y, serial, data, timestamp]
    Npts = npts[0] * npts[1]
    results = np.zeros((Npts, 7))

    # build scan order (with optional serpentine)
    scan_points = []
    for j, y in enumerate(ylist):
        row = [(i, j, x, y) for i, x in enumerate(xlist)]
        if serpentine and (j % 2 == 1):
            row.reverse()
        scan_points.extend(row)
    natural_order = np.empty(Npts, dtype=int)
    
    datafilehandle = open(filename, 'w')   
    datafilehandle.write(f"# Optical Power Meter Scan Data\n")
    datafilehandle.write("#Target X(um)\tActual X(um)\tTarget Y(um)\tActual Y(um)\tSerial\tPower(W)\tTime(s)\n")
        
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
            cmap='viridis',vmin=0, vmax=100, animated=True
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Power (W)')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('Real-Time Optical Power Map')
        
        # Force a clean initial draw, then cache the static background elements
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(ax.bbox)
        plt.show(block=False)

    t0 = time.perf_counter()
    cury = None  # Initialize current y position
    texttime = 0

    for idx, (ix, iy, x, y) in enumerate(scan_points):
        pt_start = time.perf_counter()
        # Move X and Y stages
        pxerror,movingx = devices[0].move_abs(x)
        x_time = time.perf_counter() - pt_start
        if idx == 0:            
            lastpt_start = pt_start
            ptoptime = 0
            cury = y
            pyerror,movingy = devices[1].move_abs(y)
        else:
            ptoptime = lastpt_start - pt_start
            lastpt_start = pt_start
            if y != cury:
                pyerror,movingy = devices[1].move_abs(y)
                cury = y
        y_time = time.perf_counter() - x_time - pt_start
        time.sleep(wait_time)
        measx = devices[0].get_pos()  # Ensure the position is updated before measuring
        measy = devices[1].get_pos()  # Ensure the position is updated before measuringgit

        val = measure_func()
        meas_time = time.perf_counter() - x_time - pt_start - y_time
        pt_time = time.perf_counter() - pt_start
        timestamp = time.time()
        results[idx] = (x, measx, y, measy, idx + 1, val, timestamp)
        natural_order[iy * npts[0] + ix] = idx

        datafilehandle.write(f"{x:.3f}\t{measx:.3f}\t{y:.3f}\t{measy:.3f}\t{idx + 1}\t{val:.6e}\t{timestamp}\n")
        # Update live plot and verbose print as before
        if live_plot:
            Z[iy, ix] = val
            im.set_data(Z)
            measured = Z[np.isfinite(Z)]
            im.set_clim(vmin=measured.min(), vmax=measured.max())

            # 3. Restore the cached background pixels (Instantly wipes old frame artifacts)
            fig.canvas.restore_region(background)

            # 4. Redraw ONLY the image object on top of the cached background
            ax.draw_artist(im)

            # 5. Push just the image bounding box region to the screen canvas
            fig.canvas.blit(ax.bbox)
            #fig.canvas.flush_events()
            plt.pause(0.005)
            #fig.canvas.draw()
            #fig.canvas.flush_events()
        plottime = time.perf_counter() - pt_time - pt_start
        if verbose:
            print(f"[{idx + 1}/{Npts}] x={x:.3f} measx={measx:.3f} y={y:.3f} measy={measy:.3f} errx={pxerror:.3f} erry={pyerror:.3f} | Time={pt_time:.2f}s ptop = {ptoptime:.2f}s xtime = {x_time:.2f}s ytime = {y_time:.2f}s meastime = {meas_time:.2f}s plotime = {plottime:.2f}s textime = {texttime:.2f}s → {val:.4e} W")
        texttime = time.perf_counter() - pt_time - pt_start - plottime
    total_time = time.perf_counter() - t0
    print(f"Scan complete: {Npts} pts in {total_time:.1f}s")

    # return to start
    if reset:
        devices[0].move_abs(starts[0])
        devices[1].move_abs(starts[1])

    # finalize the plot
    if live_plot:
        ax.set_title('Final Optical Power Map')
        im.set_data(Z)
        measured = Z[np.isfinite(Z)]
        im.set_clim(vmin=measured.min(), vmax=measured.max())
        im.set_animated(False)
        fig.canvas.draw();  fig.canvas.flush_events()
        plt.ioff()
    datafilehandle.close()
    results = results[natural_order]
    complete_filename = f"{os.path.splitext(filename)[0]}_complete{os.path.splitext(filename)[1]}"
    with open(complete_filename, 'w') as complete_file:
        complete_file.write("# Optical Power Meter Scan Data\n")
        complete_file.write("#Target X(um)\tActual X(um)\tTarget Y(um)\tActual Y(um)\tSerial\tPower(W)\tTime(s)\n")
        for row in results:
            complete_file.write(
                f"{row[0]:.3f}\t{row[1]:.3f}\t{row[2]:.3f}\t{row[3]:.3f}\t"
                f"{int(row[4])}\t{row[5]:.6e}\t{row[6]}\n"
            )
    os.remove(filename)
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
    x_stage.closedloop(True)
    y_stage.enable()
    y_stage.closedloop(True)
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
    xdomain = [0, 10]
    ydomain = [0, 10]
    pts = [11, 11]

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    foldertstamp = datetime.now().strftime("%Y_%m_%d")+'\\' 
    text_file_name = f"{timestamp}_optical_power_meter_scan.txt"
    folder = r"C:\\Code\\QuantumOpticsINRS\\Data\\"
    folder = os.path.join(folder, foldertstamp)
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, text_file_name)
    
    # run the sweep
    data_array, xlist, ylist = sweep_multi(
        devices=[x_stage, y_stage],
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_power_reading,
        filename=file_name,
        wait_time=0.05,
        serpentine=True,
        verbose=True,
        live_plot=True
    )

    # disable axes
    # x_stage.disable()
    # y_stage.disable()

    complete_file_name = (
        f"{os.path.splitext(file_name)[0]}_complete{os.path.splitext(file_name)[1]}"
    )
    plot_name = os.path.join(folder, f"{timestamp}_optical_power_meter_scan.png")
    plt.savefig(plot_name, dpi=300)
    # save to text file (4 columns: X, Y, Serial, Power)
    

    print(f"Results saved to {complete_file_name}")

    # close power meter / VISA session
    power_meter.close()
    rm.close()

    plt.show()

    return complete_file_name, data_array


if __name__ == "__main__":
    run_full_scan()