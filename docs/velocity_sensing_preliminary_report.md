# Reading speed and direction from a superconducting sensor sheet

### Preliminary report

**Status: simulation only.** Nothing has been built or measured. Every
performance figure below comes from a computer model of the system; every
device figure (detector quality, wire properties, amplifier speed) comes from
published measurements by other groups. The purpose of this report is to say
whether the idea is worth building, and what it would cost.

**Headline: the idea works in simulation, one serious problem was found and
solved, and the main remaining unknown is a question for you, not for us** —
namely what physical thing the sensor is meant to be watching. See §7.

---

## 1. What we are trying to do

A particle crosses a flat sheet of sensors. It touches each sensor slightly
later than the last — about **10 picoseconds** later, which is ten trillionths
of a second. For scale, light travels about 3 millimetres in that time.

From nothing but that sequence of arrival times, we want to recover **how fast
the particle was going and in what direction** — and ideally its whole path,
not just an average.

The sensors are superconducting single-photon detectors, which have to be kept
below about 4 degrees above absolute zero. The rest of the circuitry has to live
in that same cold environment, which rules out ordinary electronics and means
everything must be built from superconducting parts.

The obvious approach — read every sensor separately and compare timestamps —
does not scale. A modest 128 × 128 sheet has over sixteen thousand sensors, and
each wire coming out of a cold chamber carries heat in. We need the sheet to
work out the answer itself and report only a summary.

---

## 2. The approach, in plain terms

**Every sensor carries a tiny clock.** Each site holds a small electrical
circuit that rings at a steady rate — ten billion times a second, so one full
turn of the clock hand every 100 picoseconds. When a particle triggers that
sensor, it stamps whatever position the hand happens to be in at that instant.

**A moving particle leaves a staircase.** If the particle reaches each sensor
10 picoseconds after the last, each sensor's hand is stamped a little further
round than its neighbour's — about a tenth of a turn. Laid out across the sheet,
these stamps form a regular staircase pattern. **The steepness of that staircase
is the speed, and the direction it climbs is the heading.** Recovering velocity
becomes a matter of measuring the slope of a pattern, which is a much easier
problem than comparing thousands of timestamps.

**Tuned networks read the slope.** Connect each sensor to its neighbours through
short lengths of wire that deliberately delay the signal. If the delay in the
wire happens to match the particle's travel time between those sensors, the
signals arrive in step and reinforce each other. If it does not match, they
arrive out of step and cancel. Build many such networks side by side, each wired
for a different speed and direction, and simply see which one lights up.

This is not a new principle. It is how a barn owl locates a mouse in the dark:
sound reaches one ear fractionally before the other, and the owl's brain runs
the signals down delay lines into cells that fire only when both arrive at once.
We are building the same mechanism out of superconducting wire.

**A useful consequence:** the tuning of a network depends only on its delay
matching the particle's travel time. It does not depend on how fast the clocks
tick. That means one physical wiring layout works at any clock rate, and the
clock rate can be chosen freely to trade measurement range against precision.

**The payoff on wiring:** instead of sixteen thousand individual sensor
readouts, the sheet reports which of about 192 networks responded — roughly an
85-fold reduction in wires leaving the cold chamber. The computation replaces
the wiring.

---

## 3. What we found

Working at a clock rate of ten billion ticks per second, a 10-picosecond step
between sensors sits comfortably in the middle of the range the method handles
well — not at an awkward edge. The system covers roughly 8 to 40 picoseconds per
sensor in one configuration.

In an idealised sheet with perfect sensors, the method recovers the
10-picosecond step to better than **0.2%**. That precision is not the realistic
expectation; real performance is set by sensor timing quality, covered below.

The design turned out to be **remarkably tolerant of things going wrong**:

| what breaks | how much it can take |
|---|---|
| sensors that register nothing at all | up to **95%** of them dead |
| manufacturing spread in the delay wires | **20%** variation |
| uniform manufacturing error | unlimited — one calibration number removes it |

The tolerance to dead sensors is worth dwelling on. It means the sheet does not
need every sensor to fire, which matters because these detectors respond to
individual particles of light and will often see nothing at all. A sparse,
patchy set of hits is enough.

**One free improvement.** The obvious way to read the answer is to report
whichever tuned network responded most strongly. Reading *between* the networks
instead — fitting a smooth curve through the strongest few and finding its peak
— is about **38 times more accurate** and costs nothing in hardware. It is
purely a change in how the output is interpreted.

Getting this right took two attempts. The natural way to do the fit treats speed
and direction separately, and that turns out to be systematically biased: it
consistently under-reads the speed of anything travelling between two of the
tuned directions. Fitting both together removes the bias.

---

## 4. The problem that nearly stopped it

Early work modelled the sheet as a single line of sensors, because that is much
cheaper to simulate. That model said the system would tolerate about
25 picoseconds of timing imprecision in the sensors, which is comfortably within
what real detectors achieve.

**The full two-dimensional model said 3 picoseconds** — better than any detector
that exists. The simplified model had been over-optimistic by more than a factor
of ten.

We checked whether this was an artefact of how we were reading the answer, by
re-running with perfect knowledge of where the track actually was. It made no
difference. The problem was real.

**The cause is geometric.** A particle track is a thin line. In a
two-dimensional sheet, each sensor on that line is surrounded mostly by sensors
that the particle never touched. When we connect a sensor to all its neighbours,
most of those connections bring in nothing but noise, watering down the signal.
In a single line of sensors this never happens, because both neighbours are
always on the track. This is exactly the kind of thing a simplified model hides.

---

## 5. The fix

If the problem is that connections point at empty space, then point them along
the track instead.

Each tuned network already knows which direction it is looking for. So instead
of connecting each sensor to a circular patch of neighbours, connect it to an
elongated one — stretched along that network's own preferred direction, narrow
across it. Every connection then lands on the track rather than beside it.

**It works, and it is cheaper than the alternative:**

| design | timing tolerance needed | wiring per sensor |
|---|---|---|
| small circular connections | 3 ps — nothing available meets this | 0.5 mm |
| large circular connections | ~15 ps, marginal | 16.6 mm |
| **elongated connections** | **~25 ps** | **9.5 mm** |

The elongated design is **14 times more accurate** than the cheap circular one
under realistic timing noise, and it beats the expensive circular one while
using little more than half the wire. It also behaves far more consistently
across different particle speeds, where the circular design falls apart on the
fastest targets.

This moves the requirement from *"no detector in the literature is good enough"*
to *"several are, with margin to spare."*

**The cost is curved tracks.** An elongated connection pattern assumes the
particle travels in a straight line. On curved paths accuracy degrades — still
within tolerance in everything we tested, but it is a real trade, and it argues
against making the pattern too narrow.

---

## 6. What the hardware would need

We reviewed published measurements for each component.

**Detectors — workable, but the choice of material matters more than expected.**
Timing quality varies enormously between detector types, from about 15
picoseconds for the best niobium-nitride devices to nearly 200 picoseconds for
some alternatives. With the improved design, the best two categories work with
margin and a third is usable for coarse measurements. This is a first-order
design decision, not a detail to settle later.

**Delay wires — the component that makes this practical.** Superconducting wires
of a particular kind slow signals to a few percent of the speed of light, so a
10-picosecond delay fits in about a tenth of a millimetre of wire instead of
several millimetres. Signal loss over the distances involved is negligible. A
bonus we did not expect: these wires can be tuned electrically after
manufacture, which means calibration could be done in software rather than
locked in at fabrication.

An early estimate of how much wire would be needed was **3 to 10 times too
optimistic**, because we had assumed better wire properties than anyone has
actually published. The corrected figures are in the table above.

**Amplifiers — must be kept out of the timing path.** The standard
superconducting amplifier for this job adds up to 60 picoseconds of timing
uncertainty, which alone exceeds the entire budget. Fortunately it is not
needed: the detector's own output pulse already carries the arrival time in a
form the ringing circuit can pick up directly, with no amplifier in between. The
amplifier still has a role further downstream where its speed does not matter.

**Sensor spacing — the main physical constraint.** All that delay wire has to
fit somewhere. With the recommended design the sensors need to be about **3
millimetres apart**. That is unusually far apart for this type of detector and
is the single most demanding requirement the design imposes.

---

## 7. What we do not know

**The biggest open question is for you.** We have not been told what the sensor
is actually watching, and the answer changes the required sensor spacing by a
factor of ten thousand:

| what is being detected | spacing needed for 10 ps steps |
|---|---|
| a light pulse arriving at a shallow angle | 3 millimetres or more |
| a fast light ion | about 14 micrometres |
| a heavy molecule | a fraction of a micrometre |

The design needs roughly 3 millimetres. That happens to match the light-pulse
case almost exactly — which is either fortunate or a coincidence we should not
lean on. For anything smaller, the delay wiring would have to move to a separate
layer stacked beneath the sensors.

There is a way to sidestep this entirely: add a deliberate, known delay to each
sensor that increases steadily across the sheet. That converts *any* crossing
speed into the range the system handles best, and costs about a tenth of a
millimetre of wire per sensor. It would make the sensor layout and the network
design independent choices rather than coupled ones. We have not modelled it
yet.

**Curved tracks combined with timing noise.** We tested curvature and timing
noise separately, and the design handles each. We did not test them together,
and the elongated connection pattern is precisely the thing that should struggle
with the combination. This is the first item for the next stage.

**Manufacturing uniformity.** The design tolerates 20% variation in the delay
wires, which is loose. But no published source we found states what uniformity
is actually achieved across a wafer. This is cheap to resolve — it is a
measurement, not a literature search.

**Nothing has been built.** All of the above is simulation. The next physical
step is deliberately small: two sensors and one delay wire, to measure how much
timing uncertainty the real signal chain adds. If that comes in under about 25
picoseconds, the rest of the programme is worth pursuing.

---

## 8. Status and recommendation

A staged plan runs from the current simulation through progressively larger
hardware: two sensors, then a line of sixteen, then a line with several tuned
networks, then a full two-dimensional sheet. Each stage has a defined pass
criterion so the programme can be stopped cheaply if it fails.

The first stage is complete. Of seven checks defined in advance, **six passed.**
The one failure is the small-circular-connection design under realistic timing
noise — retained deliberately in the test suite as the documented baseline that
motivated the fix in §5.

**Recommendation: proceed**, with two caveats. Settle what is being sensed
before committing to a sensor layout, since it drives the spacing by four orders
of magnitude. And treat the curvature-plus-noise combination as a gate before
any hardware is committed, because it is the one place the current design has a
known theoretical weakness that has not been measured.

A general lesson from this stage is worth recording: **three separate
conclusions from the simplified one-dimensional model did not survive the move
to two dimensions**, and one of them was severe enough to have killed the
programme had it not been caught. Simplified models should be treated as sources
of hypotheses here, not of specifications.

---

*Technical detail, derivations, source citations and reproducible figures:
`docs/wavesheet_hardware_ladder.md`. Simulation code:
`demos/wave_velocity_bank_rung0.jl`.*
