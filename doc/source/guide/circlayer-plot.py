import cv2
import matplotlib.pyplot as plt
from circlayer import CircLayer
from circsubstrate import CircSubst

from dipcoatimage.finitedepth import Reference, get_data_path

gray = cv2.imread(get_data_path("ref2.png"), cv2.IMREAD_GRAYSCALE)
_, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ref = Reference(im, (200, 50, 1200, 200), (500, 250, 900, 600))
subst = CircSubst(ref, parameters=CircSubst.ParamType(1.0, 20.0, 100.0, 10.0))
subst.verify()

gray = cv2.imread(get_data_path("coat2.png"), cv2.IMREAD_GRAYSCALE)
_, im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
coat = CircLayer(im, subst)
plt.imshow(coat.draw())
