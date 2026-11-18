#! bin/usr/env python

import fitz
import numpy as np
import matplotlib.pyplot as plt

def parse_table(text):
    """Return an (n, 3) array of [x, Q3, q3] from one CILAS report page."""
    start = next(i for i, s in enumerate(text) if s.startswith("Standards classes")) + 1
    tokens = [t.strip() for t in text[start:]]

    rows, i = [], 0
    while i + 2 < len(tokens):
        if tokens[i:i + 3] == ["x", "Q3", "q3"]:    # repeated column header
            i += 3
            continue
        try:
            rows.append([float(t) for t in tokens[i:i + 3]])
        except ValueError:                           # reached the legend at the bottom
            break
        i += 3
    return np.array(rows)

def get_field(text, label):
    """Return the float value following a header label, e.g. 'Diameter at 50%'."""
    tokens = [t.strip() for t in text]
    i = tokens.index(label)
    for t in tokens[i + 1:i + 5]:          # value is within a few tokens of the label
        if t.startswith(":"):
            return float(t.lstrip(":").strip())
    raise ValueError(f"No value found after '{label}'")

def ReadData(file_in: str):
    doc = fitz.open(file_in)
    standard_class_headers = ['0.04', '0.07', '0.10', '0.20', '0.30', '0.40',
                              '0.50', '0.60', '0.70', '0.80', '0.90', '1.00',
                              '1.10', '1.20', '1.30', '1.40', '1.60', '1.80',
                              '2.00', '2.20', '2.40', '2.60', '3.00', '4.00',
                              '5.00', '6.00', '6.50', '7.00', '7.50', '8.00',
                              '8.50', '9.00', '10.00', '11.00', '12.00',
                              '13.00', '14.00', '15.00', '16.00', '17.00',
                              '18.00', '19.00', '20.00', '22.00', '25.00',
                              '28.00', '32.00', '36.00', '38.00', '40.00',
                              '45.00', '50.00', '53.00', '56.00', '63.00',
                              '71.00', '75.00', '80.00', '85.00', '90.00',
                              '95.00', '100.0', '106.0', '112.0', '125.0',
                              '130.0', '140.0', '145.0', '150.0', '160.0',
                              '170.0', '180.0', '190.0', '200.0', '212.0',
                              '242.0', '250.0', '300.0', '400.0', '500.0',
                              '600.0', '700.0', '800.0', '900.0', '1000.0',
                              '1100.0', '1200.0', '1300.0', '1400.0', '1500.0',
                              '1600.0', '1700.0', '1800.0', '1900.0', '2000.0',
                              '2100.0', '2200.0', '2300.0', '2400.0', '2500.0']
    standard_class_headers_float = [float(val) for val in standard_class_headers]
    samples = []
    for page_num in range(1, len(doc), 2):
        text = doc[page_num].get_text().splitlines()

        name = text[text.index("Sample Name") + 1].lstrip(": ").strip()
        x, Q3, q3 = parse_table(text).T
        assert len(x) == 100 and np.allclose(x, standard_class_headers_float), \
                f"Unexpected table on page {page_num}"

        samples.append({
        "name":   name,
        "x":      x,
        "Q3":     Q3,
        "q3":     q3,
        "mean":   get_field(text, "Mean diameter"),     # µm
        "median": get_field(text, "Diameter at 50%"),   # µm
        })
    return samples
    
def USDAClasses(discrete_cdf_list, bins, clay_max=4, silt_max=63):
    i_clay = np.argmin(np.abs(bins - clay_max))
    i_silt = np.argmin(np.abs(bins - silt_max))

    Q3_clay = discrete_cdf_list[:, i_clay]    # % finer than clay_max
    Q3_silt = discrete_cdf_list[:, i_silt]    # % finer than silt_max

    clay = Q3_clay
    silt = Q3_silt - Q3_clay
    sand = 100 - Q3_silt
    return sand, silt, clay
    
def PhiPercentile(Q3, x, p):
    d_um = np.array([10 ** np.interp(100 - p, cdf, np.log10(x)) for cdf in Q3])
    return -np.log2(d_um / 1000)      # µm -> mm -> phi

def Percentile(discrete_cdf, bins, perc):
    for i, val in enumerate(discrete_cdf):
        if val > perc:
            slope = (discrete_cdf[i]-discrete_cdf[i-1])/(bins[i]-bins[i-1])
            y_diff = perc - discrete_cdf[i-1]
            return (y_diff/slope) + bins[i-1]

def FolkWardStats(Q3, x):
    phi = {p: PhiPercentile(Q3, x, p) for p in (5, 16, 25, 75, 84, 95)}
    sorting = (phi[84] - phi[16]) / 4 + (phi[95] - phi[5]) / 6.6
    kurtosis = (phi[95] - phi[5]) / (2.44 * (phi[75] - phi[25]))
    return sorting, kurtosis


def Summarize(samples):
    x = np.array(samples[0]['x'])
    names = np.array([s['name'] for s in samples[::2]])
    q3 = np.array([np.mean([samples[i]['q3'], samples[i+1]['q3']], axis=0) for i in range(0, len(samples), 2)])
    Q3 = np.array([np.mean([samples[i]['Q3'], samples[i+1]['Q3']], axis=0) for i in range(0, len(samples), 2)])
    means = np.array([np.mean([samples[i]['mean'], samples[i+1]['mean']]) for i in range(0, len(samples), 2)])
    d10 = np.array([Percentile(cdf, x, 10) for cdf in Q3])
    d50 = np.array([Percentile(cdf, x, 50) for cdf in Q3])
    d90 = np.array([Percentile(cdf, x, 90) for cdf in Q3])
    sand, silt, clay = USDAClasses(Q3, x)
    sorting, kurtosis = FolkWardStats(Q3, x)

    depths = np.array([int(n.split('_')[1].split('-')[0]) for n in names])

    # ---------- Plot 1: distributions with depth ----------
    plt.figure(figsize=(4, 8))
    for depth, data in zip(depths, q3):
        plt.plot(x, -3 * data + depth, '-k')
    plt.title('Grain Size Distributions With Depth')
    plt.xlabel('Grain Size [μm]')
    plt.ylabel('Density Function + Core Depth [cm]')
    plt.xscale('log')
    plt.gca().invert_yaxis()

    # ---------- Plot 2: composite profile ----------
    fig, axs = plt.subplots(1, 4, figsize=(12, 10), sharey=True)

    # (0) grain size statistics
    axs[0].plot(means, depths, '-k', linewidth=1, label='Mean')
    axs[0].plot(d10, depths, linewidth=1, label='D10')
    axs[0].plot(d50, depths, linewidth=1, label='D50')
    axs[0].plot(d90, depths, linewidth=1, label='D90')
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Grain Size [μm]')
    axs[0].set_ylabel('Core Depth [cm]')
    axs[0].legend()
    axs[0].grid(alpha=0.4)

    # (1) sand / silt / clay
    sand_top = sand
    silt_top = sand + silt
    clay_top = sand + silt + clay
    axs[1].fill_betweenx(depths, 0,        sand_top, color='#e0b25c', label='Sand')
    axs[1].fill_betweenx(depths, sand_top, silt_top, color='#9aa7b0', label='Silt')
    axs[1].fill_betweenx(depths, silt_top, clay_top, color='#7a5c48', label='Clay')
    axs[1].plot(sand_top, depths, '-k', linewidth=0.5)
    axs[1].plot(silt_top, depths, '-k', linewidth=0.5)
    axs[1].set_xlim(0, 100)
    axs[1].set_xlabel('%')
    axs[1].legend(loc='lower right')

    # (2) sorting (Folk & Ward inclusive graphic standard deviation, phi)
    axs[2].plot(sorting, depths, '-ok', linewidth=1, markersize=3)
    axs[2].set_xlabel('Sorting, σ$_I$ [φ]')
    axs[2].grid(alpha=0.4)

    # (3) kurtosis (Folk & Ward graphic kurtosis, K_G)
    axs[3].plot(kurtosis, depths, '-ok', linewidth=1, markersize=3)
    axs[3].axvline(1.0, color='r', linestyle='--', linewidth=0.8, label='Mesokurtic')
    axs[3].set_xlabel('Kurtosis, K$_G$')
    axs[3].legend()
    axs[3].grid(alpha=0.4)

    axs[0].invert_yaxis()            # shared y: inverts all four panels
    fig.tight_layout()
    plt.show()

    return {"names": names, "depths": depths, "means": means, "d10": d10,
            "d50": d50, "d90": d90, "sand": sand, "silt": silt, "clay": clay,
            "sorting": sorting, "kurtosis": kurtosis}

