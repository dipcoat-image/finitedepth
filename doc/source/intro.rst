============
Introduction
============

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

For this, DipCoatImage-FiniteDepth defines three kind of classes:

1. Substrate reference
2. Substrate
3. Coating layer
