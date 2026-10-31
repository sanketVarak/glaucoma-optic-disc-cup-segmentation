import numpy as np
from src.glaucoma_seg.vcdr import largest_component, vcdr, vertical_extent


def test_vertical_extent_and_vcdr():
    disc = np.zeros((20, 20), dtype=bool)
    cup = np.zeros_like(disc)
    disc[2:18, 3:17] = True
    cup[6:14, 7:13] = True
    assert vertical_extent(disc) == 16
    assert vertical_extent(cup) == 8
    assert vcdr(disc, cup) == 0.5


def test_largest_component_keeps_largest_region():
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:8, 1:8] = True
    mask[12:14, 12:14] = True
    result = largest_component(mask)
    assert result.sum() == 49
    assert not result[12:14, 12:14].any()
