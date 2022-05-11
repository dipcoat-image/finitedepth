=========================
Class with Generic Typing
=========================

To subclass base class...

1. Annotate the base class
2. Do not use TypeAlias

To subclass concrete class, make sure that type variables are compatible, i.e. ``Subclass.Parameters`` is subclass of ``Superclass.Parameters``.
