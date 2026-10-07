import os
import numpy as np
import matplotlib.pyplot as plt

def AMC_plot(folder,file_path,maptype):
    """
    Reads the scan_output.txt in `folder`, reshapes the data into a 2D map,
    and displays it with imshow().
    """

    # load the data (skip the single header line)
    # Columns: X(um), Y(um), Serial, Data
    data = np.loadtxt(file_path, delimiter="\t", skiprows=1)
    x = data[:, 0]
    y =file_path data[:, 1]
    counts = data[:, 3]

    # determine grid dimensions
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nX, nY = x_unique.size, y_unique.size

    # reshape flat counts into a 2D array
    Z = counts.reshape((nY, nX))

    # plot
    plt.figure(figsize=(8, 6))
    plt.imshow(
        Z,
        extent=[x_unique.min(), x_unique.max(),
                y_unique.min(), y_unique.max()],
        origin='lower',
        aspect='auto',
        cmap=maptype
    )
    plt.colorbar(label='Counts')
    plt.xlabel('X (µm)')
    plt.ylabel('Y (µm)')
    plt.title("Photoluminescence map")
    plt.tight_layout()
    plt.show()


# folder = "amc_test"
# os.makedirs(folder, exist_ok=True)
# file_name = os.path.join(folder, "scan_output.txt")
#
#
# plot_pl_map(folder, file_name)