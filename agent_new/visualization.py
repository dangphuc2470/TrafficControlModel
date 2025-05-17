import matplotlib.pyplot as plt
import os
import numpy as np

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        # Convert to numpy array for easier filtering
        data_array = np.array(data)
        
        # Filter out zero values
        non_zero_mask = data_array != 0
        filtered_data = data_array[non_zero_mask]
        
        # If all values are zero, use original data
        if len(filtered_data) == 0:
            filtered_data = data_array
        
        min_val = min(filtered_data)
        max_val = max(filtered_data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        # Plot only non-zero values
        plt.plot(np.where(non_zero_mask)[0], filtered_data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)
    