===================
Designing Dataclass
===================

As shown in :doc:`defining-substrate` and other documents, user needs to define dataclasses and assign them to specific class attributes.
This document explaines the steps to design the dataclass fully compatible with API.

Defining dataclass
==================

Here, we define a nested dataclass for example.
It has :class:`enum.Enum`-type field and custom object field.

>>> from enum import Enum
>>> from dataclasses import dataclass
>>> class MyEnum(Enum):
...     a = 'a'
...     b = 'b'
>>> class MyObj:
...     def __init__(self, x: int):
...         self.x = x
...     def __repr__(self):
...         return f'MyObj(x={self.x})'
>>> @dataclass
... class MyData1:
...     x: MyEnum
...     y: MyObj
>>> @dataclass
... class MyData2:
...     md: MyData1
>>> data = MyData2(MyData1(MyEnum.a, MyObj(10)))
>>> data
MyData2(md=MyData1(x=<MyEnum.a: 'a'>, y=MyObj(x=10)))

This class alone is enough to define custom classes, but we can go further for more abilities.

Making dataclass serializable
=============================

It is a common practice to save the analysis configuration as file and load it.
Popular file format is JSON, which ensures native compatibility with python dictionary.

Let's try convert the dataclass to JSON format string.

>>> import json
>>> json.dumps(data)
Traceback (most recent call last):
 ...
TypeError: Object of type MyData2 is not JSON serializable

Serialization fails because dataclass is not directly serializable. What if we
convert it to dictionary first?

>>> from dataclasses import asdict
>>> asdict(data)
{'md': {'x': <MyEnum.a: 'a'>, 'y': MyObj(x=10)}}
>>> json.dumps(asdict(data))
Traceback (most recent call last):
 ...
TypeError: Object of type MyEnum is not JSON serializable

Serialization fails again because :class:`enum.Enum` is not directly serializable.

Of course we can resolve this by defining custom :class:`json.JSONEncoder`, but defining encoder for every field type is very tedious.
Furthermore, JSON-specific encoder cannot be used for serializing to other formats such as YAML.

Instead, we unstructure the fields to "primitive types" first and serialize them.

.. note::

   Although Python does not have primitive data type, the term is used to
   stress that return types can be parsed by other languages to their primitive
   data types.

For unstructuring, we use :mod:`cattrs`.
It is a powerful package which supports dataclass converter.
DipcoatImage-FiniteDepth provides :obj:`data_converter`, a dedicated :class:`cattrs.Converter` instance.

>>> from dipcoatimage.finitedepth import data_converter
>>> data_converter.unstructure(data)
{'md': {'x': 'a', 'y': MyObj(x=10)}}
>>> json.dumps(data_converter.unstructure(data))
Traceback (most recent call last):
 ...
TypeError: Object of type MyObj is not JSON serializable

Oops. What went wrong?
:class:`cattrs.Converter` is powerful enough to unstruture :class:`enum.Enum`, but it does not know how to convert ``MyObj``.
We have to register hooks first.

>>> data_converter.register_unstructure_hook(MyObj, lambda obj: dict(x=obj.x))
>>> data_converter.register_structure_hook(MyObj, lambda d, t: MyObj(**d))

Now we can serialize ``MyData2`` instance.

>>> unstruct_data = data_converter.unstructure(data)
>>> unstruct_data
{'md': {'x': 'a', 'y': {'x': 10}}}
>>> json.dumps(unstruct_data)
'{"md": {"x": "a", "y": {"x": 10}}}'

And we can deserialize the JSON string and structure it back to ``MyData2``!

>>> unserial_data = json.loads(json.dumps(unstruct_data))
>>> unserial_data
{'md': {'x': 'a', 'y': {'x': 10}}}
>>> data_converter.structure(unserial_data, MyData2)
MyData2(md=MyData1(x=<MyEnum.a: 'a'>, y=MyObj(x=10)))
