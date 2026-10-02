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
     - times [s] of samples or frames
   * - ``f0``
     - fundamental frequency [Hz]; 0 where unvoiced
   * - ``cf``, ``cfs``
     - centre frequency, centre frequencies of a filterbank [Hz]
   * - ``f_lo``, ``f_hi``, ``f_max``
     - band edges and upper limits [Hz]
   * - ``n_fft``
     - FFT length [samples]
   * - ``hop``
     - step between analysis frames [samples, or s where stated]
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

Counts start with ``n_`` (``n_samples``, ``n_channels``); a plural is an array
of the singular (``freqs``, ``harmonics``). Sounds are ``(n_samples,
n_channels)``: time runs down the first axis.
