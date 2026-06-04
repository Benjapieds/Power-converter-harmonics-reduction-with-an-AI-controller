import matplotlib.pyplot as plt

def init_matplotlib():
    """
    Initialize matplotlib settings for high-quality, publication-ready visualizations.
    """
    # Use a clean and professional style
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
    'figure.figsize': (8, 5),
    'figure.dpi': 300,
    'font.size': 14,  # Increased from 12 to 14
    'axes.titlesize': 16,  # Increased from 14 to 16
    'axes.labelsize': 18,  # Increased from 12 to 14
    'xtick.labelsize': 16,  # Increased from 10 to 12
    'ytick.labelsize': 16,  # Increased from 10 to 12
    'legend.fontsize': 16,  # Increased from 10 to 12
    'legend.frameon': True,                     # Show legend frame
    'legend.facecolor': 'white',                # Legend background
    'legend.edgecolor': 'black',                # Legend border color
    'legend.framealpha': 1.0,                   # Solid background
    'legend.loc': 'upper right',                 # Force legend to top right
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'savefig.transparent': True,
    'savefig.pad_inches': 0.05,
    'figure.constrained_layout.use': True,
    })

if __name__ == "__main__":
    init_matplotlib()
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    plt.plot(x, y, marker='o', label='y = x^2')
    plt.show()
