Names and conventions
=====================

Variables, locals included, are named for what they hold (``n_ramp``,
``start_phases``, ``impulse_response``); the rule and its exceptions are in
``docs/design/philosophy.md``. A few short names are the field's own and stay
short everywhere:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Name
     - Meaning
   * - ``fs``
     - sampling rate [Hz]
   * - ``t``
     - times [s] of samples or time windows
   * - ``f0``
     - fundamental frequency [Hz]; 0 where unvoiced
   * - ``cf``, ``cfs``
     - center frequency, center frequencies of a filterbank [Hz]
   * - ``f_lo``, ``f_hi``, ``f_max``
     - band edges and upper limits [Hz]
   * - ``n_fft``
     - FFT length [samples]
   * - ``hop``
     - step between time windows [samples, or s where stated]
   * - ``rms``, ``db``
     - root-mean-square level; level in decibels
   * - ``erb``
     - equivalent rectangular bandwidth, or ERB-number [Cams]
   * - ``itd``, ``ild``
     - interaural time [s] and level [dB] difference
   * - ``ir``
     - impulse response
   * - ``sos``
     - filter coefficients as second-order sections (SciPy's format)
   * - ``rng``
     - a seed or ``numpy.random.Generator``

"Frame" means only the mathematical frame (``Frame``, ``GaborFrame``,
``TVGaborFrame``: an analysis with an exact inverse). One point of an
analysis's time grid, and the stretch of sound under the window there, is a
*time window*; arrays over them have ``n_windows`` entries, and ``t`` holds
the window center times. In prose the frame classes are always set as code.

Counts start with ``n_`` (``n_samples``, ``n_channels``); a plural is an array
of the singular (``freqs``, ``harmonics``). Sounds are ``(n_samples,
n_channels)``: time runs down the first axis.
