.. _howto-define-class:

How to define your own class
============================

.. currentmodule:: finitedepth

In this document, you will learn how to build your own analysis strategy
by defining custom classes.

DipcoatImage-FiniteDepth is designed to be extensible as much as possible by
providing abstract base classes. Concrete classes such as :class:`Reference`
or :class:`RectSubstrate`, which you have encountered in previouse documents,
are their implementations. By implementing the same API, you can define your
own class that seamlessly bind with the framework.
