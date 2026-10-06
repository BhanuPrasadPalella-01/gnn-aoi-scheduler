import numpy as np


class Channel:
    """
    Time-slotted shared channel. At most `capacity` sources can be
    polled per slot. Tracks per-source Age of Information (AoI).
    """
    def __init__(self, M, capacity):
        self.M = M
        self.capacity = capacity
        self.age = np.ones(M)

    def poll(self, selected_indices):
        assert len(selected_indices) <= self.capacity, "Exceeds channel capacity"
        self.age += 1
        for i in selected_indices:
            self.age[i] = 1
        return self.age.copy()
