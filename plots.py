import matplotlib.pyplot as plt
from IPython.display import Image
import numpy as np

def plot_test_fidelity(fidelity_data, N, filepath):
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

    plt.savefig(f'{filepath}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight', dpi=300)

    # Adjust layout and display
    plt.show()
    return

#########################
#########################


def plot_test_z_expectations(z_data, N, filepath):
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


    