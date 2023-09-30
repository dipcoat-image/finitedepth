Getting started
===============

.. currentmodule:: finitedepth

What is "finite depth dip coating"?
-----------------------------------

.. figure:: ./_images/finite-depth-dip-coating.jpg
   :align: center

   The four stages of the finite-depth coating process.

Finite depth dip coating is a process in which a substrate is partially coated
with liquid by being dipped into a bath with relatively small immersion depth.
Unlike traditional dip coating, the coating layer is confined into a small
portion of the substrate and thus the lower edge effect dominates the system.

The process consists of four stages:

#. Immersion
      In this initial stage, the substrate descends into the bath.
#. Withdrawal
      The substrate is now gradually pulled out of the bath.
#. Pinch-off
      As the substrate exits the bath, a capillary bridge forms between the
      coating layer and the bulk fluid. The bridge then thins by pinch-off
      dynamics and eventually ruptures.
#. Termination
      Surface tension redistributes the fluid until equilibrium is reached.

To achieve uniform coating, the coating layer profile should be measured and
studied; and that's what *DipcoatImage-FiniteDepth* is for.

Fundamentals
------------

Image analysis is performed based on two types of images; *reference image* and
*target image*. Reference image is an image of bare substrate, and target image
is an image of coated substrate.

.. plot::
   :context: reset
   :caption: Reference image (left) and target image (right).
      Red box is substrate region.

   import cv2, numpy as np, matplotlib.pyplot as plt
   from dipcoatimage.finitedepth import *
   data = dict(
      ref_path=get_data_path("ref3.png"),
      coat_path=get_data_path("coat3.mp4"),
      reference=dict(
         templateROI=(13, 10, 1246, 200),
         substrateROI=(100, 100, 1200, 500),
         draw_options=dict(
            templateROI=dict(linewidth=0),
            substrateROI=dict(color=(255, 0, 0), linewidth=4),
         ),
      ),
      coatinglayer=dict(
         deco_options=dict(layer=dict(facecolor=(0, 0, 0))),
      ),
   )
   config = data_converter.structure(data, Config)
   coat = config.construct_coatinglayer(0, False)

   _, axes = plt.subplots(1, 2, figsize=(6, 2.5))
   axes[0].imshow(coat.substrate.reference.draw(), cmap="gray")
   axes[0].axis("off")
   axes[1].imshow(coat.draw(), cmap="gray")
   axes[1].axis("off")
   plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
   plt.show()

The basic scheme of analysis is:

#. Select the substrate region from the reference image.
#. Locate the substrate region in the target image.
#. Retrieve the coating layer region from the target image.

.. plot::
   :context: close-figs
   :caption: Coating layer region (blue) with substrate region removed.

   coat.draw_options.subtraction = coat.SubtractionMode.SUBSTRATE
   coat.deco_options.layer.facecolor = (69, 132, 182)
   plt.figure(figsize=(3, 2.5))
   plt.axis("off")
   plt.imshow(coat.draw())
   plt.tight_layout()

The resulting coating layer region can be further processed to return desired
data, e.g., thickness or unifomity.

Installation
------------

DipcoatImage-FiniteDepth can be installed by `pip` from its github repository.

.. code-block:: bash

   $ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git

This installs the package with its latest commit. If you want a specific
version, append ``@[tag name]`` such as:

.. code-block:: bash

   $ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git@v1.0.0

Basic usage
-----------

DipcoatImage-FiniteDepth provides command-line to invoke analysis using
configuration files.

.. code-block:: bash

   $ finitedepth analyze config1.yml config2.json ...

It can be run as a package as well:

.. code-block:: bash

   $ python -m dipcoatimage.finitedepth analyze config1.yml config2.json ...

The configuration file must contain entries which are constructed to
:class:`Config` instance. See :ref:`config-example` for more information.

.. note::
   To check other options, run:

   .. code-block:: bash

      $ finitedepth -h

User can also import the classes from :mod:`finitedepth` to define
their own analysis.

Next steps
----------

Check out more resources to help you customize your analysis:

* Read :ref:`tutorials` for basic examples.
