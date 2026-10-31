import cv2
import numpy as np


def vertical_extent(mask):
    # Height in pixels of the mask's bounding box
    # (first to last foreground row)
    rows = np.flatnonzero(mask.any(axis=1))

    return 0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)


def largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8
    )

    if n <= 2:
        # Empty mask or a single component
        return mask.astype(bool)

    return lab == 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )


def vcdr(disc, cup):
    d = vertical_extent(disc)

    return float("nan") if d == 0 else vertical_extent(cup) / d
