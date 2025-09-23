import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from IPython.display import Image
import numpy as np
#peak analysis
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def plot_test_fidelity(fidelity_data, N, filepath=None):
    plt.figure(figsize=(8, 6), dpi=300)
    # Plot each curve
    num_steps = len(fidelity_data)
    plt.plot(range(num_steps), fidelity_data, label=f'Fidelity', linestyle='-', linewidth=2)

    # Add labels, title, legend, and grid
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Fidelity', fontsize=14)
    plt.title(f'Fidelity Curve for N={N}')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)

    # Adjust layout and display
    plt.show()
    return

#########################
#########################


def plot_test_z_expectations(z_data, N, filepath=None):
    # Create simplified figure with only line plot
    fig = plt.figure(figsize=(8, 6), dpi=300)

    # Custom colors
    first_color = '#1f77b4'  # Blue
    last_color = '#d62728'   # Red
    middle_color = '#4a4a4a' # Dark gray

    # ========== SIMPLIFIED LINE PLOT ==========
    ax = fig.add_subplot(111)

    for i in range(N):
        magn = z_data[:,i]
        norm_time = np.linspace(0, 1, len(magn))

        lineprops = {
            'color': first_color if i == 0 else (last_color if i == N-1 else middle_color),
            'lw': 2.5 if i in [0, N-1] else 1.0,
            'alpha': 1.0 if i in [0, N-1] else 0.6,
            'label': r'First spin $(n=0)$' if i == 0 else (r'Last spin $(n={})$'.format(N-1) if i == N-1 else None)
        }

        ax.plot(norm_time, magn, **lineprops)

    # Formatting
    ax.set_xlabel(r'Normalized time $t/\tau_{\mathrm{transfer}}$', fontsize=12)
    ax.set_ylabel(r'Magnetization $\langle Z \rangle$', fontsize=12)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=10)

    # Clean grid
    ax.grid(True, linestyle=':', alpha=0.3, color='gray')

    # Simplified legend - only show first and last if many spins
    if N > 10:
        # Only show first and last for clarity
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0], handles[-1]], [labels[0], labels[-1]], 
                  fontsize=10, framealpha=0.9, loc='best')
    else:
        ax.legend(fontsize=9, framealpha=0.9, loc='best')

    # Title
    plt.title(r'Spin Magnetization Dynamics ($N={}$)'.format(N), fontsize=14, pad=15)

    # Final layout adjustment
    plt.tight_layout()

    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)

    plt.show()
    return


#########################
#########################



def plot_ratios_trend_slope(x_data, y_data, log_scale=True, show_trend=True, filepath = '../'):
    """Create publication-quality plot of error ratios with trend line."""
    plt.figure(figsize=(8, 6), dpi=300)  # High resolution for thesis
    
    # Create a figure with constrained layout
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Plot original data with improved marker styling
    ax.plot(x_data, y_data, 'b-', linewidth=2, marker='o', markersize=6, 
            markeredgecolor='k', markeredgewidth=0.5, label='Simulated data')
    
    # Calculate and plot trend line if requested
    if show_trend:
        # Linear regression using numpy
        coeffs = np.polyfit(x_data, y_data, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        trend_line = np.poly1d(coeffs)
        y_trend = trend_line(x_data)
        
        # Plot trend line with improved styling
        ax.plot(x_data, y_trend, 'r--', linewidth=1.5, 
                label=f'Linear fit (α = {slope:.3f})')

    # --- Professional Styling ---
    # Font styling (Times New Roman-like)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Axis labels with requested font sizes
    ax.set_xlabel(r'$log(J/\lambda)$', fontsize=15, labelpad=10)
    ax.set_ylabel('$\log(1 - F)$', fontsize=15, labelpad=10)
    
    # Title with requested font size
    ax.set_title('Error vs. Ratio $J/\lambda$', fontsize=16, pad=12)
    
    # Scientific notation for y-axis if log scale
    if log_scale:
        ax.set_yscale('log')
        # Format log scale ticks properly
        from matplotlib.ticker import LogFormatter
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Grid styling
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)
    
    # Spine styling - remove top/right, adjust others
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')
    #legend with professional styling
    if show_trend:
        legend = ax.legend(fontsize=12, frameon=True, 
                          framealpha=1, edgecolor='k',
                          loc='upper left' if log_scale else 'best')
        legend.get_frame().set_linewidth(0.8)
    
    # Save figure if path provided
    plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
    
    
    plt.show()
    return


########################################
########################################

def plot_multiple_ratio_trend_slope(x_data, y_data_dict, log_scale=True, show_trend=True, filepath='../'):
    """Create publication-quality plot of error ratios with trend lines for multiple curves.
    
    Args:
        x_data (list): X-axis data (same for all curves)
        y_data_dict (dict): Dictionary where keys are curve names and values are y-data lists
        log_scale (bool): Whether to use log scale on y-axis
        show_trend (bool): Whether to calculate and show trend lines
        filepath (str): Path to save the figure
    """
    # Create a figure with constrained layout
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Define a color palette for multiple curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_data_dict)))
    
    # Store handles and labels for legend
    legend_handles = []
    legend_labels = []
    
    # Plot each curve
    for i, (curve_name, y_data) in enumerate(y_data_dict.items()):
        color = colors[i]
        
        # Plot original data with improved marker styling - store the handle
        line, = ax.plot(x_data, y_data, '-', linewidth=2, marker='o', markersize=6, 
                       markeredgecolor='k', markeredgewidth=0.5, color=color)
        
        # Calculate and store trend line if requested
        if show_trend:
            # Linear regression using numpy
            coeffs = np.polyfit(x_data, y_data, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            trend_line = np.poly1d(coeffs)
            y_trend = trend_line(x_data)
            
            # Plot trend line with matching color but dashed
            ax.plot(x_data, y_trend, '--', linewidth=1.5, color=color, alpha=0.8)
            
            # Use the actual data line handle for legend, but with slope in label
            legend_handles.append(line)
            legend_labels.append(f'{curve_name} (α = {slope:.3f})')
        else:
            # If no trend lines, just use curve name
            legend_handles.append(line)
            legend_labels.append(curve_name)

    # --- Professional Styling ---
    # Font styling (Times New Roman-like)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Axis labels with requested font sizes
    ax.set_xlabel(r'$log(J/\lambda)$', fontsize=15, labelpad=10)
    ax.set_ylabel('$\log(1 - F)$', fontsize=15, labelpad=10)
    
    # Title with requested font size
    ax.set_title('Error vs. Ratio $J/\lambda$', fontsize=16, pad=12)
    
    # Scientific notation for y-axis if log scale
    if log_scale:
        ax.set_yscale('log')
        # Format log scale ticks properly
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Grid styling
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)
    
    # Spine styling - remove top/right, adjust others
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')
    
    # Create legend using the actual line handles
    if legend_handles:
        legend = ax.legend(legend_handles, legend_labels, fontsize=12, frameon=True, 
                          framealpha=1, edgecolor='k',
                          loc='upper left' if log_scale else 'best')
        legend.get_frame().set_linewidth(0.8)
    
    # Save figure if path provided
    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
    
    plt.show()

    return


##################################
################################## 

def plot_multiple_ratio_trend_slope_2(x_data, y_data_dict, log_scale=True, show_trend=True, filepath='../'):
    """Create publication-quality plot of error ratios with trend lines for multiple curves.
    
    Args:
        x_data (list): X-axis data (same for all curves)
        y_data_dict (dict): Dictionary where keys are curve names and values are y-data lists
        log_scale (bool): Whether to use log scale on y-axis
        show_trend (bool): Whether to calculate and show trend lines
        filepath (str): Path to save the figure
    """
    # Create a figure with constrained layout
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    # Use the color palette from the second function (commonly used colors in quantum plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Store handles and labels for legend
    legend_handles = []
    legend_labels = []
    
    # Plot each curve
    for i, (curve_name, y_data) in enumerate(y_data_dict.items()):
        color = colors[i % len(colors)]  # Cycle through colors if more curves than colors
        
        # Plot original data with improved marker styling - store the handle
        line, = ax.plot(x_data, y_data, '-', linewidth=2, marker='o', markersize=6, 
                       markeredgecolor='k', markeredgewidth=0.5, color=color)
        
        # Calculate and store trend line if requested
        if show_trend:
            # Linear regression using numpy
            coeffs = np.polyfit(x_data, y_data, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            trend_line = np.poly1d(coeffs)
            y_trend = trend_line(x_data)
            
            # Plot trend line with matching color but dashed
            ax.plot(x_data, y_trend, '--', linewidth=1.5, color=color, alpha=0.8)
            
            # Use the actual data line handle for legend, but with slope in label
            legend_handles.append(line)
            legend_labels.append(f'{curve_name} (α = {slope:.3f})')
        else:
            # If no trend lines, just use curve name
            legend_handles.append(line)
            legend_labels.append(curve_name)

    # --- Professional Styling ---
    # Font styling (Times New Roman-like)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    
    # Axis labels with requested font sizes
    ax.set_xlabel(r'$log(J/\lambda)$', fontsize=15, labelpad=10)
    ax.set_ylabel('$\log(1 - F)$', fontsize=15, labelpad=10)
    
    # Title with requested font size
    ax.set_title('Error vs. Ratio $J/\lambda$', fontsize=16, pad=12)
    
    # Scientific notation for y-axis if log scale
    if log_scale:
        ax.set_yscale('log')
        # Format log scale ticks properly
        ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    # Grid styling
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.5)
    
    # Spine styling - remove top/right, adjust others
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('black')
    
    # Create legend using the actual line handles
    if legend_handles:
        legend = ax.legend(legend_handles, legend_labels, fontsize=12, frameon=True, 
                          framealpha=1, edgecolor='k',
                          loc='upper left' if log_scale else 'best')
        legend.get_frame().set_linewidth(0.8)
    
    # Save figure if path provided
    if filepath:
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)
    
    plt.show()

    return


##################################
##################################

def extract_peak_evolution(num_steps, fidelity_data, min_peak_height=0.2, smooth_points=200):
    """Extract peaks with fixed frequency from start until first natural peak."""
    t = np.linspace(0, num_steps-1, num_steps)
    
    # First find all natural peaks that meet height requirement
    natural_peaks, _ = find_peaks(fidelity_data, height=min_peak_height)
    natural_peak_times = t[natural_peaks]
    natural_peak_values = fidelity_data[natural_peaks]
    
    if len(natural_peaks) < 2:
        # Not enough peaks to determine frequency - return basic data
        return {
            't': t,
            'y': fidelity_data,
            'peak_times': natural_peak_times,
            'peak_values': natural_peak_values,
            'smooth_t': natural_peak_times,
            'smooth_y': natural_peak_values,
            'max_fidelity': max(natural_peak_values) if len(natural_peak_values) > 0 else np.nan
        }
    
    # Calculate peak frequency in data points
    peak_intervals = np.diff(natural_peaks)
    points_per_peak = int(round(np.mean(peak_intervals)))
    
    # Determine first natural peak position
    first_natural_peak_pos = natural_peaks[0]
    
    # Generate regularly spaced peaks from start until first natural peak
    all_peak_positions = []
    
    # Start from first peak and work backwards
    current_pos = first_natural_peak_pos
    while current_pos >= 0:
        all_peak_positions.append(current_pos)
        current_pos -= points_per_peak
    
    # Now go forward from start to first natural peak
    all_peak_positions = sorted(all_peak_positions)
    all_peak_positions = [p for p in all_peak_positions if p <= first_natural_peak_pos]
    
    # Combine with natural peaks (removing duplicates)
    combined_peaks = np.unique(np.concatenate([
        np.array(all_peak_positions),
        natural_peaks
    ]))
    
    # Get corresponding times and values
    peak_times = t[combined_peaks]
    peak_values = fidelity_data[combined_peaks]
    
    # Create smoothed trend
    if len(peak_times) > 1:
        interp_func = interp1d(peak_times, peak_values, kind='cubic', fill_value='extrapolate')
        smooth_t = np.linspace(min(peak_times), max(peak_times), smooth_points)
        smooth_y = interp_func(smooth_t)
    else:
        smooth_t, smooth_y = peak_times, peak_values
    
    return {
        't': t,
        'y': fidelity_data,
        'peak_times': peak_times,
        'peak_values': peak_values,
        'smooth_t': smooth_t,
        'smooth_y': smooth_y,
        'max_fidelity': max(peak_values) if len(peak_values) > 0 else np.nan
    }


##################################
##################################

def plot_three_fidelity_curves(num_steps, fidelity_data_list, labels=None, colors=None, min_peak_height=0.0, filepath = None):
    """PhD-quality plot of fidelity trends with peak evolution."""
    
    # Default styling (Nature-style colors)
    if labels is None:
        labels = ['Set 1', 'Set 2', 'Set 3']
    if colors is None:
        colors = ['#4E79A7', '#F28E2B', '#59A14F']  # Muted blue, orange, green
    
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True, dpi=300)
    
    # First pass to find the time of maximum for the orange line (index 1)
    orange_data = extract_peak_evolution(len(fidelity_data_list[1]), fidelity_data_list[1], min_peak_height)
    max_time_orange = orange_data['smooth_t'][np.argmax(orange_data['smooth_y'])]
    scaling_factor = 2 / (max_time_orange / (num_steps-1))  # This will make max_time_orange correspond to 2
    
    # Process and plot each dataset
    for i, fidelity_data in enumerate(fidelity_data_list):
        data = extract_peak_evolution(len(fidelity_data), fidelity_data, min_peak_height)
        
        # Rescale time axis so orange maximum is at 2
        rescaled_t = data['t'] / (num_steps-1) * scaling_factor
        rescaled_peak_times = data['peak_times'] / (num_steps-1) * scaling_factor
        rescaled_smooth_t = data['smooth_t'] / (num_steps-1) * scaling_factor
        
        # Original data (very faint)
        ax.plot(rescaled_t, data['y'], color=colors[i], alpha=0.3, linewidth=0.8, zorder=1)
        
        # Peaks (no legend entry)
        ax.scatter(rescaled_peak_times, data['peak_values'], 
                  color=colors[i], s=25, alpha=0.7, zorder=2, 
                  edgecolors='k', linewidths=0.3)
        
        # Smoothed trend (with max value in legend)
        ax.plot(rescaled_smooth_t, data['smooth_y'], 
               color=colors[i], linewidth=2.5, 
               label=fr'{labels[i]} ($F_{{\max}} = {data["max_fidelity"]:.3f}$)',
               zorder=3)
    
    # Professional formatting with requested modifications
    ax.set_xlabel(r'Time ($t/\tau_{\mathrm{transfer}}$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Fidelity', fontsize=14, labelpad=10)
    ax.tick_params(labelsize=14)
    
    # Legend without peaks
    legend = ax.legend(frameon=False, framealpha=1, 
                      loc='upper left',
                      borderpad=0.8, handlelength=1.5,
                      fontsize=12)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_linewidth(0.5)
    
    # Grid and spines
    ax.grid(True, linestyle=':', color='lightgray', alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(0.7)
    
    ax.set_ylim(-0.05, 1.05)
    # Set xlim based on the scaling factor (0 to 2*scaling_factor might not be appropriate)
    # Instead, find the maximum time across all datasets after rescaling
    max_x = max([max(data['t'] / (num_steps-1)) * scaling_factor for data in 
                [extract_peak_evolution(len(fd), fd, min_peak_height) for fd in fidelity_data_list]])
    ax.set_xlim(0, max_x)
    
    # Save as vector graphic for publications
    
    if filepath:        
        plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)

    plt.show()
    return 