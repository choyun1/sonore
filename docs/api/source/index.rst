sonore API reference
====================

Every public name in sonore, generated from the docstrings. The pages follow
the package's layout: the trunk bottom to top (core, signals, frames, views),
each building only on the ones before it, then the branches (spatial, stimuli,
texture), which build on the trunk and never on each other (see `the layout notes
<https://github.com/choyun1/sonore/blob/main/docs/design/layout.md>`_).
In code, everything here is also reachable as ``so.<name>`` after
``import sonore as so``.

The `listening gallery <https://choyun1.github.io/sonore/gallery/>`_ shows
these objects at work.

.. note::

   This reference is being written alongside a line-by-line human audit of the
   code, one module at a time (`issue #48
   <https://github.com/choyun1/sonore/issues/48>`_). Until a module's sitting
   is ticked off there, its page shows the docstrings as they were first
   written, and some names have none yet.

.. toctree::
   :maxdepth: 2

   conventions
   core
   signals
   frames
   views
   spatial
   stimuli
   texture
   plotting

.. toctree::
   :caption: Elsewhere

   Listening gallery <https://choyun1.github.io/sonore/gallery/>
   sonore on GitHub <https://github.com/choyun1/sonore>
