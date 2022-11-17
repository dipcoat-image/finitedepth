============
Introduction
============

.. currentmodule:: dipcoatimage.finitedepth

DipCoatImage-FiniteDepth is a Python package to perform image analysis on the coating layer shape from the batch dip coating process with finite depth.

The image below shows how the finite depth dip coating is performed.

.. figure:: ./_images/finite-depth-dip-coating.jpg
   :align: center

   Finite depth dip coating process; immersion, deposition, termination and fluid redistribution.

As the substrate is immersed into the bulk fluid and then drawn out, liquid layer is applied onto the substrate.
The termination of coating is characterised by the lower end effect of the layer, where the capillary bridge forms and soon ruptures.
After the coating layer is separated from the pool, capillary force makes the layer to change its shape by fluid redistribution.

Below are the images of bare substrate and coated substrate from acutal coating process.

.. plot::
   :context: reset
   :caption: Bare substrate image, coated substate image
   :align: center

   import cv2, matplotlib.pyplot as plt
   from dipcoatimage.finitedepth import get_samples_path

   ref_path = get_samples_path("ref3.png")
   ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
   coat_path = get_samples_path("coat3.png")
   coat_img = cv2.imread(coat_path, cv2.IMREAD_GRAYSCALE)

   _, axes = plt.subplots(1, 2, figsize=(8, 4))
   axes[0].imshow(ref_img, cmap="gray")
   axes[0].axis("off")
   axes[1].imshow(coat_img, cmap="gray")
   axes[1].axis("off")
   plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
   plt.show()

From the coated substrate image, coating layer region can be extracted for analysis.

.. plot::
   :context: close-figs
   :caption: Coating layer region from the coated substrate image
   :align: center

   from dipcoatimage.finitedepth import SubstrateReference, Substrate, LayerArea

   ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
   ref = SubstrateReference(ref_img, (50, 50, 1200, 200), (200, 50, 1000, 600))
   subst = Substrate(ref)
   coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
   coat = LayerArea(coat_img, subst)
   coat.draw_options.remove_substrate = True

   plt.axis("off")
   plt.imshow(coat.draw())

For this, :mod:`dipcoatimage.finitedepth` defines three kind of classes:

1. Substrate reference
2. Substrate
3. Coating layer

Substrate reference
===================

Substrate reference class specifies the template ROI and substrate ROI from the bare substrate image.

.. plot::
   :context: reset
   :include-source:
   :caption: Template ROI (green), substrate ROI (red)
   :align: center

   >>> import cv2
   >>> from dipcoatimage.finitedepth import get_samples_path, SubstrateReference
   >>> import matplotlib.pyplot as plt #doctest: +SKIP

   >>> ref_path = get_samples_path("ref3.png")
   >>> ref_img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
   >>> templateROI, substrateROI = (50, 50, 1200, 200), (200, 30, 1000, 600)
   >>> ref = SubstrateReference(ref_img, templateROI, substrateROI)
   >>> plt.imshow(ref.draw()) #doctest: +SKIP

Substrate
=========

Substrate class analyzes the geometery of the bare substrate.

.. plot::
   :context: close-figs
   :include-source:
   :caption: Edge of the substrate (blue) detected by :class:`.RectSubstrate`
   :align: center

   >>> from dipcoatimage.finitedepth import CannyParameters, HoughLinesParameters, RectSubstrate

   >>> cparams = CannyParameters(50, 150)
   >>> hparams = HoughLinesParameters(1, 0.01, 50)
   >>> params = RectSubstrate.Parameters(cparams, hparams)
   >>> subst = RectSubstrate(ref, parameters=params)
   >>> subst.draw_options.draw_lines = False
   >>> plt.imshow(subst.draw()) #doctest: +SKIP

Coating layer
=============

Coating layer class extracts the coating layer by locating the bare substrate using template matching.
It then analyzes the coating layer image using the geometery from the substrate class.

.. plot::
   :context: close-figs
   :include-source:
   :caption: Coating layer region (blue) extracted by :class:`.LayerArea`
   :align: center

   >>> from dipcoatimage.finitedepth import LayerArea

   >>> coat_path = get_samples_path("coat3.png")
   >>> coat_img = cv2.cvtColor(cv2.imread(coat_path), cv2.COLOR_BGR2RGB)
   >>> coat = LayerArea(coat_img, subst)
   >>> coat.draw_options.draw_substrate = False
   >>> coat.analyze()
   LayerAreaData(Area=41747)
   >>> plt.imshow(coat.draw()) #doctest: +SKIP

GUI
===

:mod:`dipcoatimage.finitedepth_gui` provides GUI which wraps :mod:`dipcoatimage.finitedepth`.

The following code runs the GUI.

.. code-block:: python

   from PySide6.QtWidgets import QApplication
   import sys
   from dipcoatimage.finitedepth_gui import AnalysisGUI

   app = QApplication(sys.argv)
   window = AnalysisGUI()
   window.show()
   app.exec()
   app.quit()
