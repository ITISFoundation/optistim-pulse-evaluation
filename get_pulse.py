import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np

## dont wwanna integrate this changes in S4LNF and have to build a new wheel
## therefore, keep using local file
from stimulation_pulse import StimulationPulse

DST = 100  ## number of time points in a ms -- must match (or be a divisor) of w GAFCalculator DST
SEGMENT_PW = 1 / DST  # Width, in ms, of 1 pulse segment
DURATION = 1.0


def get_sinusoidal_pulse(*args) -> np.ndarray:
    """From a series of A,B coeffs, generate a sum of sinusoids and cosines"""
    assert len(args) % 2 == 0, "Number of arguments must be even."
    N = len(args) // 2
    assert N > 0, "At least one coefficient must be provided."
    A = args[:N]
    B = args[N:]
    assert len(A) == len(B), "Number of A and B coefficients must match."

    print("To enforce zero net current, A0 and B0 will be set to 0.")
    A[0] = 0.0
    B[0] = 0.0

    time_vector = np.linspace(0, DURATION, int(DURATION * DST))
    amps = np.zeros_like(time_vector)
    for j, (a, b) in enumerate(zip(A, B)):
        omega = 2 * np.pi * j / DURATION
        amps += a * np.cos(omega * time_vector) + b * np.sin(omega * time_vector)

    return amps


def get_pulse(*args, stds=None) -> StimulationPulse:

    pulse_object = StimulationPulse(None)
    pulse_object.name = "Sum-of-Sinusoids Pulse"

    current_balanced_amplitudes = get_sinusoidal_pulse(*args)

    if stds is not None:
        assert len(stds) == len(current_balanced_amplitudes), (
            "Number of stds must match the number of amplitudes."
            + f"Currently {len(stds)} and {len(current_balanced_amplitudes)}"
        )
        ## TODO would I need to propagate the stds through the sinusoidal function?
        # how to do that?
        if not np.all(np.array(stds) == 0):
            raise NotImplementedError("stds not implemented yet for sinusoidal pulses")

    # Create the pulse
    for i, amp in enumerate(current_balanced_amplitudes):
        std = stds[i] if stds is not None else None
        pulse_object._insert_time_interval(amp, SEGMENT_PW, std=std)

    pulse_object.finish_pulse(DURATION)
    if stds is None:
        pulse_object.std_list = None

    return pulse_object


if __name__ == "__main__":
    pulse = get_pulse(*list(np.random.randn(30)))  # TESTING
    pulse.plot_pulse(show=True)
    print("Done")
