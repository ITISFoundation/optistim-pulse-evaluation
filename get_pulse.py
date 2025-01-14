import os
import sys

sys.path.append(os.path.dirname(__file__))

from stimulation_pulse import StimulationPulse
import numpy as np

SEGMENT_PW = 0.1  # Width, in ms, of 1 pulse segment
DURATION = 1.0


def create_zero_mean_projection(xN1):
    """Projects to a one-higher dimension in which the mean is always zero.
    This should help make the inputs more independently meaninginful (instead of substracting the mean, depending on all others).
    However, I fear it might be equivalent to substracting the mean.
    In that case, simply try the alternative, appending a compensating pulse at the end, or simply removing the demeaning condition.
    """

    N = len(xN1) + 1
    ones_N = np.ones((N, 1))

    P = np.eye(N) - 1 / N * (ones_N @ ones_N.T)
    xN = P @ np.hstack([xN1, [0.0]])
    return xN


def get_pulse(
    *args, segment_pw: float = SEGMENT_PW, duration=DURATION, stds=None
) -> StimulationPulse:
    assert len(args) == (int(duration / segment_pw) - 1), (
        "Number of arguments must match the duration of the pulse."
        + f"Currently {len(args)} and {int(duration/segment_pw)-1}"
    )

    pulse_object = StimulationPulse(None)
    pulse_object.name = "Free Pulse"

    amps = np.array(args)
    # current_balanced_amplitudes = create_zero_mean_projection(amps)
    # we will substract the mean from the pulse, to make it charge-balanced
    current_balanced_amplitudes = (
        amps  # no current balance -- check if SuMo gets better
    )

    # Create the pulse
    for i, amp in enumerate(current_balanced_amplitudes):
        std = stds[i] if stds is not None else None
        pulse_object._insert_time_interval(amp, segment_pw, std=std)

    pulse_object.finish_pulse(DURATION)

    return pulse_object
