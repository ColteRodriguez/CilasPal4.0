#! bin/usr/env python

import fitz
import numpy as np
import matplotlib.pyplot as plt

def ParseTable(text):
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

def ReadData(file_in):
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
    for page_num in range(1, len(doc), 2):               # every second page; use range(1, 2, 2) for just page 1
        text = doc[page_num].get_text().splitlines()
        name = text[text.index("Sample Name") + 1].lstrip(": ").strip()
        x, Q3, q3 = ParseTable(text).T
        # sanity check against the known class boundaries
        assert len(x) == 100 and np.allclose(x, standard_class_headers_float), \
                f"Unexpected table on page {page_num}"
        samples.append({"name": name, "x": x, "Q3": Q3, "q3": q3})

    return samples

