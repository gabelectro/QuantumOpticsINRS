import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
from AMC import Device
from Swabian import TimeTagger


class AMC_Axis:
    """Wrapper for piezo stage axis control"""
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

    def get_pos(self):
        return self.dev.move.getPosition(self.axis_id) / 1e3


def sweep_multi_optimized(devices, starts, stops, npts, measure_func,
                          reset=True, wait_time=0.05,
                          serpentine=False, verbose=False,
                          live_plot=False, print_interval=50):
    """
    Optimized scanning function with reduced overhead.
    
    Key optimizations:
    - Batched console output (only print every print_interval points)
    - Live plotting disabled by default (use final plot instead)
    - Configurable wait_time (default 0.05s instead of 0.1s)
    """
    
    xlist = np.linspace(starts[0], stops[0], npts[0])
    ylist = np.linspace(starts[1], stops[1], npts[1])

    Npts = npts[0] * npts[1]
    results = np.zeros((Npts, 4))

    # Build scan order
    scan_points = []
    for j, y in enumerate(ylist):
        row = [(x, y) for x in xlist]
        if serpentine and (j % 2 == 1):
            row.reverse()
        scan_points.extend(row)

    # Initialize live plot if requested (not recommended; slows down acquisition)
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
        cbar.set_label('Counts')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_title('Real‑Time PL Map')
        plt.show()

    t0 = time.perf_counter()
    
    # Batch timing info
    batch_times = []

    for idx, (x, y) in enumerate(scan_points):
        pt_start = time.perf_counter()
        
        # Move stages
        devices[0].move_abs(x)
        devices[1].move_abs(y)
        time.sleep(wait_time)

        # Measure
        val = measure_func()

        pt_time = time.perf_counter() - pt_start
        results[idx] = (x, y, idx + 1, val)
        batch_times.append(pt_time)

        # Update live plot (expensive, only if explicitly requested)
        if live_plot:
            ix = int(round((x - xlist[0]) / (xlist[-1] - xlist[0]) * (npts[0] - 1)))
            iy = int(round((y - ylist[0]) / (ylist[-1] - ylist[0]) * (npts[1] - 1)))
            Z[iy, ix] = val
            im.set_data(Z)
            im.autoscale()
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Batched verbose output (print every print_interval points)
        if verbose and (idx + 1) % print_interval == 0:
            elapsed = time.perf_counter() - t0
            avg_pt_time = np.mean(batch_times[-print_interval:])
            remaining_pts = Npts - (idx + 1)
            est_remaining = remaining_pts * avg_pt_time
            print(f"[{idx + 1}/{Npts}] Elapsed: {elapsed:.1f}s | "
                  f"Avg: {avg_pt_time:.3f}s/pt | Est. remaining: {est_remaining:.1f}s")

    total_time = time.perf_counter() - t0
    print(f"\nScan complete: {Npts} points in {total_time:.1f}s ({total_time/Npts*1000:.1f}ms/point)")

    # Return to start
    if reset:
        devices[0].move_abs(starts[0])
        devices[1].move_abs(starts[1])

    # Finalize live plot if enabled
    if live_plot:
        ax.set_title('Final PL Map')
        plt.ioff()
        plt.show()

    return results, xlist, ylist


def run_full_scan_optimized():
    """
    Optimized version of run_full_scan with faster execution.
    
    Changes:
    - wait_time reduced from 0.1s to 0.05s (test and adjust if needed)
    - live_plot disabled (final plot generated after scan)
    - verbose output batched (only every 50 points by default)
    """
    
    # Collect metadata from user
    laser_power_uw = float(input("Laser power (uW): "))
    focus = input("Focus level: ")
    filter_used = input("Filter used: ")
    count_before_filter = input("Count before filter: ")
    count_after_filter = input("Count after filter: ")
    nw_diameter = input("nw diameter: ")

    # Connect to hardware
    print("Connecting to AMC100 controller...")
    amc = Device("192.168.1.1")
    amc.connect()
    print("Connected.")
    
    x_stage = AMC_Axis(amc, axis_id=1)
    y_stage = AMC_Axis(amc, axis_id=2)

    x_stage.enable()
    y_stage.enable()
    print("Stages enabled.")

    # Set up TimeTagger
    print("Initializing TimeTagger...")
    tagger = TimeTagger.createTimeTagger()
    tagger.setTriggerLevel(channel=1, voltage=1.0)
    tagger.setTriggerLevel(channel=2, voltage=1.0)
    counter = TimeTagger.Counter(
        tagger=tagger,
        channels=[1, 2],
        binwidth=int(2e10),
        n_values=10
    )
    print("TimeTagger ready.")

    def get_APD_counts():
        """Measure APD counts for 0.2 seconds"""
        duration_s = 0.2
        duration_ps = int(duration_s * 1e12)
        counter.startFor(capture_duration=duration_ps)
        counter.waitUntilFinished()
        data = counter.getData()
        return int(data[0].sum())

    # Scan parameters
    xdomain = [-7, 7]
    ydomain = [-7, 7]
    pts = [35, 35]

    print(f"\nStarting scan: {pts[0]}×{pts[1]} grid ({pts[0]*pts[1]} points)")
    print(f"X range: {xdomain[0]} to {xdomain[1]} µm")
    print(f"Y range: {ydomain[0]} to {ydomain[1]} µm")
    print("(Scan in progress...)\n")

    # Run optimized sweep
    data_array, xlist, ylist = sweep_multi_optimized(
        devices=[x_stage, y_stage],
        starts=[xdomain[0], ydomain[0]],
        stops=[xdomain[1], ydomain[1]],
        npts=pts,
        measure_func=get_APD_counts,
        wait_time=0.1,        # OPTIMIZED: Reduced from 0.1
        serpentine=True,
        verbose=True,          # Show batched progress
        live_plot=False,       # OPTIMIZED: Disable live plot (MAJOR speedup)
        print_interval=100      # Print every 50 points
    )

    # Disable stages
    x_stage.disable()
    y_stage.disable()
    print("Stages disabled.")

    # Generate filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    text_file_name = (
        f"{timestamp}_{nw_diameter}_{laser_power_uw}_{focus}_"
        f"{filter_used}_{count_before_filter}_{count_after_filter}_PL_Map_data.txt"
    )
    folder = r"C:\\Student Folders\\Iftekhar\\Aug 2026\\After Gabriel Alignment"
    os.makedirs(folder, exist_ok=True)
    file_name = os.path.join(folder, text_file_name)

    # Save data
    print(f"\nSaving data to {file_name}...")
    header = "X(um)\tY(um)\tSerial\tData\n"
    np.savetxt(
        file_name,
        data_array,
        fmt=["%0.3f", "%0.3f", "%d", "%d"],
        delimiter="\t",
        header=header,
        comments=""
    )
    print("Data saved.")

    # Generate final plot (once, not during scan)
    print("Generating final plot...")
    nX, nY = len(xlist), len(ylist)
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
    plt.title('PL Map')
    plt.tight_layout()
    
    png_file = os.path.join(folder, text_file_name.replace(".txt", ".png"))
    plt.savefig(png_file, dpi=600, format="png")
    print(f"Plot saved to {png_file}")
    plt.show()

    print(f"\n✓ Scan complete!")
    print(f"  Data:  {file_name}")
    print(f"  Image: {png_file}")
    
    return file_name, data_array


if __name__ == "__main__":
    run_full_scan_optimized()
