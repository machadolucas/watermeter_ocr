"""Shared deterministic offline LCD capture/scheduling simulation."""
import random

from watermeter import ViewPhase


def simulate(true_period, cycles=400, phi=3.0, lead=1.4, interval=20.0,
              phase_factory=ViewPhase, timing_jitter=0.0, processing_sec=0.8):
    """Three views, capture lead, post-exposure work and main's deadline.

    `phase_factory` permits an offline comparison against the old class;
    jitter models edge/lead variation without changing accepted readings.
    Every cycle, including acquisition, must still get a numeric reading.
    """
    p = phase_factory(20.0, 11.0, margin_sec=1.0, capture_lead_sec=lead)
    rng = random.Random(1)
    t = 0.0
    first_miss = no_reading = 0
    for cycle in range(cycles):
        start = p.next_start(t) if p.confident() else None
        if start is None:
            start = p.explore_start(t)
        tt = start
        got = False
        for attempt in range(3):
            actual_lead = lead + rng.uniform(-timing_jitter, timing_jitter)
            e = tt + actual_lead
            off = (e - phi + rng.uniform(-timing_jitter, timing_jitter)) % true_period
            num = off < 11.0
            cells = list("__40106__" if off < 14.9 else "_6_1_426_")
            p.note_lead(actual_lead)
            p.observe(e, num, cells=None if num else cells)
            done = e + processing_sec
            if cycle >= 20 and attempt == 0 and not num:
                first_miss += 1
            if num:
                got = True
                break
            nxt = p.next_start(done) if p.confident() else None
            tt = done + (min(max(0.0, nxt - done), 20.0) if nxt is not None else 5.5)
        if not got:
            no_reading += 1
        t = max(start + interval, done)
    return first_miss / (cycles - 20), no_reading
