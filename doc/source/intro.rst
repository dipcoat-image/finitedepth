============
Introduction
============

.. currentmodule:: dipcoatimage.finitedepth

DipCoatImage-FiniteDepth is a Python package to perform image analysis on the coating layer shape from batch dip coating process with finite depth.

Dip coating with finite immersion depth is commonly used to apply liquid film onto a three-dimensional object.
The image below shows how the finite depth dip coating is performed.

.. figure:: ./_images/finite-depth-dip-coating.jpg
   :align: center

   Finite depth dip coating process; immersion, deposition, termination and fluid redistribution.

The termination of the process is goverened by the lower end effect of the system, where the capillary bridge is formed between the bulk fluid and the coating layer and soon ruptures.
After the capillary bridge breaks, coating layer changes its shape over time by fluid redistribution.
This temporal evolution of the coating layer must be analyzed to optimize the coating process.

Analysis images
===============

Two silhouette images are required to analyze the coating layer shape:

1. Bare substrate image
2. Coated substrate image

Below are the images of the bare substrate and the coated substrate from acutal coating process.

.. plot::
   :context: reset
   :caption: Bare substrate image(left) and coated substate image (right)
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

From these two images, the coating layer region can be extracted and further analyzed to yield the quantitative data (i.e., coating layer thickness).
Temporal evolution of the coating layer can be assessed by analyzing the series of coated substrate images from the coating process.

.. plot::
   :context: close-figs
   :caption: Coating layer region image
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

Analysis classes
================

:mod:`dipcoatimage.finitedepth` defines three kind of classes for image analysis:

1. Substrate reference
2. Substrate
3. Coating layer

Substrate reference
-------------------

Substrate reference class is a container for the bare substrate image and two ROIs; template ROI and substrate ROI.

The first ROI specifies the template region for the coating layer class, and the second specifies the substrate region for the substrate class.

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

:class:`.SubstrateReferenceBase` is an abstract base class of substrate reference classes.
Every substrate reference class must be its implementation by subclassing it.

Substrate
---------

Substrate class detects the geometry of the substrate.
It uses the bare substrate image cropped by the substrarte reference instance.

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

:class:`.SubstrateBase` is an abstract base class of substrate classes.
Every substrate class must be its implementation by subclassing it.

Coating layer
-------------

Coating layer class extracts the coating layer region from the coated substrate image and the bare substrate image, and then retrieves quantitative data.

To extract the coating layer region, it performs template matching between the template region from the substrate reference instance and the coated substrate image.
To analyze the coating layer shape, it uses the substrate geometry information detected by the substrate instance.

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

:class:`.CoatingLayerBase` is an abstract base class of coating layer classes.
Every coating layer class must be its implementation by subclassing it.

GUI
===

:mod:`dipcoatimage.finitedepth_gui` provides the GUI to perform visualization and analysis.

Its main features are:

1. Construct the classes from :mod:`dipcoatimage.finitedepth` by specifying the parameters.
2. Visualize the constructed classes.
3. Save and load the serialized parameters.
4. Real-time visualization of the image stream from the camera.
5. Capturing the image and recording the video from the camera.
6. Analyzing the experiment using constructed classes.

The following code runs the GUI.
Style sheet is set to highlight the mandatory field widgets.

.. code-block:: python

   from PySide6.QtWidgets import QApplication
   import sys
   from dipcoatimage.finitedepth_gui import MainWindow

   app = QApplication(sys.argv)
   app.setStyleSheet("*[requiresFieldValue=true]{border: 1px solid red}")
   window = MainWindow()
   window.show()
   app.exec()
   app.quit()
