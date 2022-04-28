==
CI
==

Dependencies
============

CI runs in headless environment, but it should be able to test GUI features.
For this, additional dependencies need to be installed in CI environment.

Python package
--------------

pytest-xvfb is required to run pytest with virtual X frame buffer. [1]_

External program
----------------

For xvfb, these packages need to be installed : ``libxkbcommon-x11-0``,
``libxcb-icccm4`,` ``libxcb-image0``, ``libxcb-keysyms1``, ``libxcb-randr0``,
``libxcb-render-util0``, ``libxcb-xinerama0``, ``libxcb-xfixes0`` [1]_

Qt6 (hence PySide6) breaks libopengl in headless environment, so ``freeglut3``
and ``freeglut3-dev`` should be manually installed in CI workflow [2]_

For ``PySide6.QtMultimedia``, gstreamer needs to be installed with these
packages : ``libgstreamer1.0-dev``, ``libgstreamer-plugins-base1.0-dev``,
``gstreamer1.0-plugins-base``, ``gstreamer1.0-plugins-good``,
``gstreamer1.0-plugins-bad``, ``gstreamer1.0-plugins-ugly``,
``gstreamer1.0-libav`` [3]_

.. [1] https://pytest-qt.readthedocs.io/en/latest/troubleshooting.html

.. [2] https://stackoverflow.com/questions/65751536

.. [3] https://gstreamer.freedesktop.org/documentation/installing/on-linux.html
