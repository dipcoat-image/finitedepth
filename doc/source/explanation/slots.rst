__slots__
=========

Classes in DipcoatImage-FiniteDepth intentionally not define
:obj:`~object.__slots__`.

Slots are useful when large number of objects simultaneously exist,
which is unlikely in our case where each object consumes large memory
because of the image data it stores. Instead of constructing multiple
objects, we need to handle small number of objects and destroy them
as soon as possible. Hence, having slots does more harm than good.
