import numpy as np 
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up Nature-compatible figure parameters
# Nature requires vector graphics (PDF/EPS) with typeface sizes 5-7pt
# Set global parameters for Nature-ready figures
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],  # Nature prefers Arial
    'font.size': 7,             # Base font size 7pt for most text
    'axes.labelsize': 8,        # Slightly larger for axis labels
    'axes.titlesize': 8,        # Same for titles
    'xtick.labelsize': 7,       # Tick labels
    'ytick.labelsize': 7,
    'legend.fontsize': 7,       # Legend text
    'figure.dpi': 300,          # High resolution for display
    'savefig.dpi': 600,         # Even higher for saving
    'savefig.bbox': 'tight',    # Tight bounding box
    'savefig.pad_inches': 0.05, # Minimal padding
    'axes.linewidth': 0.5,      # Thinner axes lines
    'grid.linewidth': 0.5,      # Thinner grid lines
    'lines.linewidth': 1.0,     # Standard line width
    'lines.markersize': 3,      # Smaller markers
    'xtick.major.width': 0.5,   # Thinner ticks
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'axes.xmargin': 0.02,       # Tighter margins
    'axes.ymargin': 0.02,
})

# Nature journal column width is 89mm, full page width is 183mm
# Standard single-column width in inches (89mm)
SINGLE_COLUMN = 89/25.4  # Convert mm to inches
# Standard double-column width in inches (183mm)
DOUBLE_COLUMN = 183/25.4  # Convert mm to inches
# Maximum height for a figure (250mm)
MAX_HEIGHT = 250/25.4  # Convert mm to inches

# Define colors that work well for print (CMYK-friendly)
# Using a colorblind-friendly palette
COLORS = {
    'blue': '#0072B2',       # Blue
    'orange': '#E69F00',     # Orange
    'green': '#009E73',      # Green
    'red': '#D55E00',        # Red/Vermilion
    'purple': '#CC79A7',     # Pink/Purple
    'yellow': '#F0E442',     # Yellow
    'grey': '#999999',       # Grey
    'black': '#000000'       # Black
}


gamma = 2 * np.pi * 4258  # rad / (s * gauss), gyromagnetic ratio

def rotation_x(theta):
    # left-handed rotation matrix about x-axis
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])


def rotation_y(theta):
    # left-handed rotation matrix about y-axis
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]])


def rotation_z(theta):
    # left-handed rotation matrix about z-axis
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def Bloch_simulation(M, b1, w_rf, w_off, dt, N_ro):
    """
    Bloch-simulate BS frequency encoding
    
    Parameters:
    -----------
    M: ndarray
        Initial magnetization
    b1: ndarray
        RF field in gauss
    w_rf: float
        RF frequency in radians/second
    w_off: float
        Frequency offset in radians/second
    dt: float
        Time step in seconds
    N_ro: int
        Number of readout samples
        
    Returns:
    --------
    M_xy: ndarray
        Complex transverse magnetization in Larmor frame
    M_z: ndarray
        Longitudinal magnetization
    M_xy_rotating_frame: ndarray
        Complex transverse magnetization in rotating frame (w_off)
    """
    M_bloch = M.copy()
    N_x = b1.size

    theta = np.arctan2(gamma * b1, w_rf)

    # assume adiabatic transition to spin-lock
    for i in range(N_x):
        R_y = rotation_y(-theta[i])
        M_bloch[:, i] = R_y @ M_bloch[:, i]

    # apply pre-phasor
    R_z = rotation_z((-w_rf + w_off) * dt)
    R_x = np.zeros((N_x, 3, 3))
    for i in range(N_x):
        R_x[i, :, :] = rotation_x(-gamma * dt * b1[i])
    for i in range(N_ro // 2):
        # apply the z-rotation
        M_bloch = R_z @ M_bloch
        # apply the x-rotation
        M_bloch = np.einsum('ijk,ki->ji', R_x, M_bloch)

    # apply readout and record the transverse magnetization
    M_xy = np.zeros((N_ro, N_x), dtype=complex)
    M_z = np.zeros((N_ro, N_x))
    R_z = rotation_z((w_rf + w_off) * dt)
    R_x = np.zeros((N_x, 3, 3))
    for i in range(N_x):
        R_x[i, :, :] = rotation_x(gamma * dt * b1[i])
    for i in range(N_ro):
        # store the transverse magnetization
        M_xy[i, :] = M_bloch[0, :] + 1j * M_bloch[1, :]
        # store the longitudinal magnetization
        M_z[i, :] = M_bloch[2, :]
        # apply the x-rotation
        M_bloch = np.einsum('ijk,ki->ji', R_x, M_bloch)
        # apply the z-rotation
        M_bloch = R_z @ M_bloch

    # demodulate to move back to Larmor frame
    M_xy_rotating_frame = M_xy.copy()
    M_xy = np.exp(1j * w_rf * np.arange(N_ro_sim) * dt_sim)[:, None] * M_xy

    return M_xy, M_z, M_xy_rotating_frame


def fftc(x):
    # centered fft
    return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(x)))


def ifftc(x):
    # centered ifft
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(x)))


# generate M0 for a two-sphere phantom setup 
N_x = 1024
fov = 0.045  # m, fov of domain
dxyz = fov / N_x  # m, resolution
x = (np.arange(N_x) - N_x // 2) * dxyz
x_center_1 = -0.0125  # m, center of first sphere
x_center_2 = 0.0125  # m, center of second sphere
r = 0.01  # m, radius of balls

# Calculate the volume of the cylinder cross section of the sphere at each x-location
M0 = np.zeros(N_x)  # equilibrium magnetization
for i in range(N_x):
    if x[i] >= x_center_1 - r and x[i] <= x_center_1 + r:
        r_cylinder = np.sqrt(r ** 2 - (x[i] - x_center_1) ** 2)
        M0[i] = 2 * np.pi * r_cylinder ** 2 * dxyz  # m^3, volume of cylinder cross section
    if x[i] >= x_center_2 - r and x[i] <= x_center_2 + r:
        r_cylinder = np.sqrt(r ** 2 - (x[i] - x_center_2) ** 2)
        M0[i] = 2 * np.pi * r_cylinder ** 2 * dxyz
M0 /= M0.max()


flip_angle = 30  # degrees, flip angle of excitation pulse
M = np.array([np.sin(np.deg2rad(flip_angle)), 0., np.cos(np.deg2rad(flip_angle))])[:, None] @ M0[None, :]  # initial magnetization

c = 0.55 ** 2 / fov  # gauss^2 / m, curvature of b1 field
b1 = np.sqrt(c * (x + fov / 2))  # Gauss, b1 field

w_rf = 2 * np.pi * 10e3  # rad/s, frequency of RF pulse
w_eff = np.sqrt((gamma * b1) ** 2 + w_rf ** 2)  # rad/s, effective field
theta = np.arctan2(gamma * b1, w_rf)  # rad, angle of effective field

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9))

# Figure 1 - Simulation Setup
ax1.plot(x * 100, M0, color='black', linewidth=1.5)
ax1.set_ylabel(r'$M_0$', size=14)
ax1.set_xticklabels([])
ax1.tick_params(axis='both', which='major', labelsize=14)

# Plot B1
ax2.plot(x * 100, b1, color='black', linewidth=1.5)
ax2.set_ylabel(r'$B_1$ (Gauss)', size=14)
ax2.set_xticklabels([])
ax2.tick_params(axis='both', which='major', labelsize=14)

# Plot tilt angle and cos(tilt angle) with separate y-axes
ax3a = ax3
ax3a.plot(x * 100, np.rad2deg(theta), 'b-', label=r'$\theta$ (degrees)', color='black', linewidth=1.5)
ax3a.set_xlabel('x (cm)', size=14)
ax3a.set_ylabel(r'$\theta$ (Degrees)', color='black', size=14)
ax3a.tick_params('y', colors='black', labelsize=14)
ax3a.tick_params(axis='both', which='major', labelsize=14)

ax3b = ax3a.twinx()
ax3b.plot(x * 100, np.cos(theta), 'r-', label=r'cos($\theta$)', linewidth=1.5)
ax3b.set_ylabel(r'cos($\theta$)', color='red', size=14)
ax3b.set_ylim(0, 1.05)
ax3b.tick_params('y', colors='red', labelsize=14)
ax3b.spines['right'].set_color('red')

fig.tight_layout()

# save the figure as a .eps
fig.savefig('bs_frequency_encoding_simsetup.eps', format='eps', dpi=600)
fig.savefig('bs_frequency_encoding_simsetup.png', format='png', dpi=600)

# Timing parameters
T_ro = 20.48e-3  # seconds, readout duration
N_ro_exp = 512  # number of readout samples for experiment
N_ro_sim = 10 * N_ro_exp  # number of readout samples for simulation, 10x finer than actual readout to minimize hard pulse approximation error
dt_exp = T_ro / N_ro_exp  # seconds, readout time step
dt_sim = T_ro / N_ro_sim  # seconds, time step

# Run the Bloch simulation
M_xy, _, _ = Bloch_simulation(M, b1, w_rf, 0., dt_sim, N_ro_sim)
# Sub-sample to experimental sampling frequency
M_xy = M_xy[::N_ro_sim // N_ro_exp, :]
# Integrate across space to get signal
signal = np.sum(M_xy, axis=1) * dxyz

# Compare to the Full Signal Equation
signal_exact = np.zeros(N_ro_exp, dtype=complex)
for i in range(N_ro_exp):
    signal_exact[i] = dxyz * np.sum(
            M[2, :] * np.sin(theta)
            + np.real((M[0, :] + 1j * M[1, :]) * np.exp(-1j * w_eff * (i - N_ro_exp // 2) * dt_exp))
            + 1j * np.imag((M[0, :] + 1j * M[1, :]) * np.exp(-1j * w_eff * (i - N_ro_exp // 2) * dt_exp)) * np.cos(theta)
        ) * np.exp(1j * w_rf * i * dt_exp)
t = (np.arange(N_ro_exp) - N_ro_exp // 2) * dt_exp  # seconds, readout time axis

fig, ax = plt.subplots(3, 2, figsize=(12, 8))

# Figure 2 - Bloch-Simulated Signal Versus Signal Equation
ax[0, 0].plot(1000 * t, np.abs(signal), label='Bloch-Simulated', color='black', linewidth=0.5)
ax[0, 0].set_ylabel('|Signal|', size=14)
ax[0, 0].set_title('Bloch Simulation', size=14)
ax[0, 0].set_ylim(0, 0.02)
ax[0, 0].tick_params(axis='both', which='major', labelsize=14)
ax[0, 0].set_xticklabels([])

ax[0, 1].plot(1000 * t, np.abs(signal_exact), label='Signal Equation', color='black', linewidth=0.5)
ax[0, 1].set_title('Signal Equation', size=14)
ax[0, 1].set_ylim(0, 0.02)
# turn off visibility of y axis labels
ax[0, 1].tick_params(axis='both', which='major', labelsize=14)
ax[0, 1].set_xticklabels([])
ax[0, 1].set_yticklabels([])

# Plot real part of signal
ax[1, 0].plot(1000 * t, np.real(signal), label='Bloch-Simulated', color='black', linewidth=0.5)
ax[1, 0].set_ylabel('Re{Signal}', size=14)
ax[1, 0].tick_params(axis='both', which='major', labelsize=14)
ax[1, 0].set_ylim(-0.02, 0.02)
ax[1, 0].set_xticklabels([])

ax[1, 1].plot(1000 * t, np.real(signal_exact), label='Signal Equation', color='black', linewidth=0.5)
ax[1, 1].set_ylim(-0.02, 0.02)
ax[1, 1].tick_params(axis='both', which='major', labelsize=14)
ax[1, 1].set_xticklabels([])
ax[1, 1].set_yticklabels([])

# Plot imaginary part of signal
ax[2, 0].plot(1000 * t, np.imag(signal), label='Bloch-Simulated', color='black', linewidth=0.5)
ax[2, 0].set_xlabel('Time (ms)', size=14)
ax[2, 0].set_ylabel('Im{Signal}', size=14)
ax[2, 0].tick_params(axis='both', which='major', labelsize=14)
ax[2, 0].set_ylim(-0.02, 0.02)

ax[2, 1].plot(1000 * t, np.imag(signal_exact), label='Signal Equation', color='black', linewidth=0.5)
ax[2, 1].set_xlabel('Time (ms)', size=14)
ax[2, 1].tick_params(axis='both', which='major', labelsize=14)
ax[2, 1].set_yticklabels([])
ax[2, 1].set_ylim(-0.02, 0.02)

fig.tight_layout()

# save the figure as a .eps
fig.savefig('bs_frequency_encoding_simversusequation.eps', format='eps', dpi=600)
fig.savefig('bs_frequency_encoding_simversusequation.png', format='png', dpi=600)

# The Reconstruction

# Display spectrum and filter out w_rf component
signal_filt = fftc(signal)

# make a low-pass filter with a tukey window
f_ro = np.fft.fftshift(np.fft.fftfreq(N_ro_exp, dt_exp))
H = tukey(2 * int(np.ceil(np.sum(np.abs(f_ro) < 7.5e3) / 2)), 0.75)
H = np.pad(H, (N_ro_exp - len(H)) // 2)

# Figure 3 - The signal spectrum before and after filtering
# fig, ax1 = plt.subplots()

# ax1.plot(f_ro, np.abs(signal_filt), label='Before Filtering')
# ax1.plot(f_ro, np.abs(signal_filt * H), label='After Filtering')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.set_ylabel('|Signal|')
# ax1.legend(loc='upper left')

# ax2 = ax1.twinx()
# ax2.plot(f_ro, H, 'r-', label='Filter')
# ax2.set_ylabel('|H|', color='r')
# ax2.tick_params('y', colors='r')


fig3 = plt.figure(figsize=(SINGLE_COLUMN, 0.7*SINGLE_COLUMN))
ax1 = fig3.add_subplot(111)

# Plot spectrum before and after filtering
ax1.plot(f_ro/1000, np.abs(signal_filt), lw=0.5, color=COLORS['blue'], label='Before Filtering')
ax1.plot(f_ro/1000, np.abs(signal_filt * H), lw=0.5, color=COLORS['green'], label='After Filtering')
ax1.set_xlabel('Frequency (kHz)')
ax1.set_ylabel('|Signal| (a.u.)')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Plot filter as a semi-transparent fill
alpha_fill = 0.2
ax2 = ax1.twinx()
ax2.fill_between(f_ro/1000, H, alpha=alpha_fill, color=COLORS['red'])
ax2.plot(f_ro/1000, H, lw=1, color=COLORS['red'], label='Filter')
ax2.set_ylabel('Filter Amplitude', color=COLORS['red'])
ax2.tick_params('y', colors=COLORS['red'])
ax2.spines['top'].set_visible(False)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left', fontsize=6)

plt.tight_layout()
plt.savefig('bs_frequency_encoding_filtering.eps', format='eps', dpi=600)
plt.savefig('bs_frequency_encoding_filtering.png', format='png', dpi=600)


# Apply the Filter
signal_filt = (signal_filt * H[None, :]).flatten()
signal_filt = ifftc(signal_filt)


# Compare to Approximate Fourier Signal Equation
G = gamma * c / (2 * w_rf)  # gauss / m, effective gradient strength
x_ro = 2 * np.pi * f_ro / (gamma * G)  # m, x-axis
center_freq = gamma * G * fov / 2
signal_filt_shift = np.exp(1j * center_freq * t) * signal_filt

signal_approx = dxyz * np.exp(-1j * gamma * G * t.reshape((N_ro_exp, 1)) @ x.reshape((1, N_x))) @ (M[0, :] + 1j * M[1, :])
phase_shift = np.angle(np.mean(signal_approx * np.conj(signal_filt_shift)))
# Compensate some global phase shift
signal_filt_shift *= np.exp(1j * phase_shift)

# Figure 4 - Time-domain signal post-filtering versus Fourier signal
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN / 3))

# Plot real part of the signal
ax1.plot(1000 * t, np.real(signal_filt_shift), color=COLORS['blue'], label='Signal Post-Filtering')
ax1.plot(1000 * t, np.real(signal_approx), color=COLORS['green'], label='Fourier Signal')
ax1.set_ylabel('Re{Signal}')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.legend(frameon=False, loc='upper left', fontsize=6)
ax1.set_xlabel('Time (ms)')

# Plot imaginary part of the signal
ax2.plot(1000 * t, np.imag(signal_filt_shift), color=COLORS['blue'], label='Signal Post-Filtering')
ax2.plot(1000 * t, np.imag(signal_approx), color=COLORS['green'], label='Fourier Signal')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Im{Signal}')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend(frameon=False, loc='upper left', fontsize=6)

# Ensure the y-axis range is the same for both subplots
y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('bs_frequency_encoding_postfilterfourier.eps', format='eps', dpi=600)
plt.savefig('bs_frequency_encoding_postfilterfourier.png', format='png', dpi=600)


# zero-pad the signal
pad_factor = 5
signal_filt_shift_zp = np.pad(signal_filt_shift, (((pad_factor - 1) // 2) * N_ro_exp, ((pad_factor - 1) // 2) * N_ro_exp), mode='constant')
# reconstruct the zero-padded profile
f_recon = np.fft.fftshift(np.fft.fftfreq(pad_factor * N_ro_exp, dt_exp))
x_recon = 2 * np.pi * f_recon / (gamma * G)  # m, x-axis

M_xy_recon = ifftc(signal_filt_shift_zp) / dxyz * pad_factor * N_ro_exp * gamma / 2 / np.pi * G * dxyz * dt_exp

# Simulate and reconstruct an off-resonant signal
# Run the Bloch simulation
off_res = 100.  # Hz, off-resonance frequency
M_xy_offres, _, _ = Bloch_simulation(M, b1, w_rf, 2 * np.pi * off_res, dt_sim, N_ro_sim)

# sub-sample to experimental sampling frequency
M_xy_offres = M_xy_offres[::N_ro_sim // N_ro_exp, :]

# sum across space to get signal
signal_offres = np.sum(M_xy_offres, axis=1) * dxyz

# display spectrum and filter out w_rf component
signal_offres_filt_shift = np.exp(1j * center_freq * t) * ifftc(H * fftc(signal_offres))

# zero-pad the signal
signal_offres_filt_shift_zp = np.pad(signal_offres_filt_shift, (((pad_factor - 1) // 2) * N_ro_exp, ((pad_factor - 1) // 2) * N_ro_exp), mode='constant')
# reconstruct the zero-padded profile
M_xy_offres_recon = ifftc(signal_offres_filt_shift_zp) / dxyz * pad_factor * N_ro_exp * gamma / 2 / np.pi * G * dxyz * dt_exp

# Simulate and reconstruct signal with larger B1 gradient
c_large = 4 * 0.55 ** 2 / fov  # gauss^2 / m, curvature of b1 field
b1_large = np.sqrt(c_large * (x + fov / 2))  # Gauss, b1 field
M_xy_large, _, _ = Bloch_simulation(M, b1_large, w_rf, 0., dt_sim, N_ro_sim)
# sub-sample to experimental sampling frequency
M_xy_large = M_xy_large[::N_ro_sim // N_ro_exp, :]
# sum across space to get signal
signal_large = np.sum(M_xy_large, axis=1) * dxyz
signal_large_filt = ifftc(H * fftc(signal_large))

G_large = gamma * c_large / (2 * w_rf)  # gauss / m, effective gradient strength
center_freq_large = gamma * G_large * fov / 2
signal_large_filt_shift = np.exp(1j * center_freq_large * t) * signal_large_filt

# zero-pad the signal
signal_large_filt_shift_zp = np.pad(signal_large_filt_shift, (((pad_factor - 1) // 2) * N_ro_exp, ((pad_factor - 1) // 2) * N_ro_exp), mode='constant')
# generate an x-axis for the entire profile
x_large_recon = 2 * np.pi * f_recon / (gamma * G_large)  # m, x-axis

# reconstruct the zero-padded profile
M_xy_large_recon = ifftc(signal_large_filt_shift_zp) / dxyz * pad_factor * N_ro_exp * gamma / 2 / np.pi * G_large * dxyz * dt_exp

# plot the profiles in a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN / 1.5))

# plot the first profile
ax1.plot(x_recon * 100, np.abs(M_xy_recon), lw=1.5, color=COLORS['blue'])
ax1.set_xlim([-fov * 100, fov * 100])
ax1.set_ylabel('|Mxy| (/M0)', fontsize=10)
ax1.set_title('Reconstructed Profile', fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_xticklabels([])  # Make x-axis labels invisible

# plot the second profile
ax2.plot(x_large_recon * 100, np.abs(M_xy_large_recon), lw=1.5, color=COLORS['green'])
ax2.set_xlim([-fov * 100, fov * 100])
ax2.set_xlabel('x (cm)', fontsize=10)
ax2.set_ylabel('|Mxy| (/M0)', fontsize=10)
ax2.set_title('2x B1 Amplitude / 4x Gradient Amplitude', fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('highgradient.eps', format='eps', dpi=600)
plt.savefig('highgradient.png', format='png', dpi=600)

# Figure: Bloch-Simulated Signal and Reconstructed Profile
# Plot Bloch-Simulated Signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, DOUBLE_COLUMN / 3))
ax1.plot(1000 * t, 100 * np.abs(signal), label='Bloch-Simulated', color='black', linewidth=0.5)
# ax1.set_xlim([])
ax1.set_xlabel('Acquisition Time (ms)', size=12)
ax1.set_ylabel('a.u.', size=12)
ax1.set_title('Signal Amplitude', size=12)
ax1.set_ylim(0, 100 * 0.02)
ax1.tick_params(axis='both', which='major', labelsize=12)
# ax1.set_xticklabels([])

# Plot Reconstructed Profile
ax2.plot(x_recon * 100, np.abs(M_xy_recon), label='Reconstructed Profile', color='black', linewidth=0.5)
ax2.set_xlim([-fov * 100, fov * 100])
ax2.set_xlabel('x (cm)', size=12)
ax2.set_ylabel('|Mxy| (/M0)', size=12)
ax2.set_title('Reconstructed Profile', size=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig('signal_recon.eps', format='eps', dpi=600)
plt.savefig('signal_recon.png', format='png', dpi=600)

fig3 = plt.figure(figsize=(SINGLE_COLUMN, 0.7*SINGLE_COLUMN))
ax1 = fig3.add_subplot(111)

# Plot On- and Off-Resonant Profiles
ax1.plot(x_recon * 100, np.abs(M_xy_recon), lw=1.5, color=COLORS['blue'], label='On-Resonance')
ax1.plot(x_recon * 100, np.abs(M_xy_offres_recon), lw=1.5, color=COLORS['green'], label=f'{off_res} Hz\nOff-Resonance')
ax1.axvline(x=1.6, color='red', linestyle='--', linewidth=1)
ax1.set_xlim([-fov * 100, fov * 100])
ax1.set_xlabel('x (cm)')
ax1.set_ylabel('|Mxy| (/M0)')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, frameon=False, loc='upper left', fontsize=6)

plt.tight_layout()
plt.savefig('bs_frequency_encoding_chemshift.eps', format='eps', dpi=600)
plt.savefig('bs_frequency_encoding_chemshift.png', format='png', dpi=600)

# compare small versus large flip angles
flip_angle_large = 90  # degrees, flip angle of excitation pulse
M_large_flip = np.array([np.sin(np.deg2rad(flip_angle_large)), 0., np.cos(np.deg2rad(flip_angle_large))])[:, None] @ M0[None, :]  # initial magnetization

# run the Bloch simulation
M_xy_large_flip, _, _ = Bloch_simulation(M_large_flip, b1, w_rf, 0., dt_sim, N_ro_sim)

# sub-sample to experimental sampling frequency
M_xy_large_flip = M_xy_large_flip[::N_ro_sim // N_ro_exp, :]

# integrate across space to get signal
signal_large_flip = np.sum(M_xy_large_flip, axis=1) * dxyz

# display spectrum and filter out w_rf component
signal_large_flip_filt_shift = np.exp(1j * center_freq * t) * ifftc(H * fftc(signal_large_flip))

# zero-pad the signal
signal_large_flip_filt_shift_zp = np.pad(signal_large_flip_filt_shift, (((pad_factor - 1) // 2) * N_ro_exp, ((pad_factor - 1) // 2) * N_ro_exp), mode='constant')

# reconstruct the zero-padded profile
M_xy_large_flip_recon = ifftc(signal_large_flip_filt_shift_zp) / dxyz * pad_factor * N_ro_exp * gamma / 2 / np.pi * G * dxyz * dt_exp

# compare small and large flip signal amplitudes, spectra, and reconstructed profiles
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

# plot the signal amplitudes
ax1.plot(1000 * t, np.abs(signal), label='30 Degree Flip', linewidth=2.0)
ax1.plot(1000 * t, np.abs(signal_large_flip), label='90 Degree Flip', linewidth=2.0)
ax1.set_xlabel('Time (ms)', fontsize=14)
ax1.set_ylabel('|Signal|', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# plot the signal spectra
ax2.plot(f_ro, np.abs(fftc(signal)), label='30 Degree Flip', linewidth=2.0)
ax2.plot(f_ro, np.abs(fftc(signal_large_flip)), label='90 Degree Flip', linewidth=2.0)
ax2.set_xlabel('Frequency (Hz)', fontsize=14)
ax2.set_ylabel('|Signal|', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# plot the reconstructed profiles
ax3.plot(x_recon * 100, np.abs(M_xy_recon), label='30 Degree Flip', linewidth=2.0)
ax3.plot(x_recon * 100, np.abs(M_xy_large_flip_recon), label='90 Degree Flip', linewidth=2.0)
ax3.set_xlim([-fov * 100, fov * 100])
ax3.set_xlabel('x (cm)', fontsize=14)
ax3.set_ylabel('|Mxy| (/M0)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.legend(fontsize=12, frameon=False)
ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.savefig('30vs90.eps', format='eps', dpi=600)
plt.savefig('30vs90.png', format='png', dpi=600)

plt.show()



print('done')
