# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=redefined-outer-name
from io import BytesIO
from io import TextIOWrapper

import pytest

from aiq.utils.type_converter import ConvertException
from aiq.utils.type_converter import GlobalTypeConverter
from aiq.utils.type_converter import TypeConverter


# --------------------------------------------------------------------
# Example classes to test inheritance-based conversions
# --------------------------------------------------------------------
class Base:

    def __init__(self, name="Base"):
        self.name = name

    def __repr__(self):
        return f"<Base name={self.name}>"


class Derived(Base):

    def __init__(self, name="Derived"):
        super().__init__(name)

    def __repr__(self):
        return f"<Derived name={self.name}>"


# --------------------------------------------------------------------
# Example converters
# --------------------------------------------------------------------


def convert_str_to_int(s: str) -> int:
    """Converts a numeric string to int."""
    try:
        return int(s)
    except ValueError:
        raise ConvertException("String is not numeric")  # pylint: disable=raise-missing-from


def convert_int_to_str(x: int) -> str:
    """Converts an integer to a string."""
    return str(x)


def convert_dict_to_str(d: dict) -> str:
    """
    Converts a dictionary to string.
    If the dict has a key "value", return that as the string
    (useful for multi-hop tests).
    """
    if "value" in d:
        return str(d["value"])
    return str(d)


def convert_str_to_float(s: str) -> float:
    """Converts a string to a float if possible."""
    try:
        return float(s)
    except ValueError:
        raise ConvertException("String cannot be converted to float")  # pylint: disable=raise-missing-from


# ----- Converters for the inheritance tests -----


def convert_base_to_str(b: Base) -> str:
    """
    Convert a Base object (or anything that inherits from Base) to a string.
    The original code review wants a direct converter:
       Base -> str
    We'll use the object's repr for demonstration.
    """
    return repr(b)


def convert_str_to_derived(s: str) -> Derived:
    """
    Convert a string to a Derived object.
    In a real scenario, you might parse the string
    or do something domain-specific.
    """
    # trivial example: store the string in the Derived's name
    d = Derived(name=f"Derived from '{s}'")
    return d


# --------------------------------------------------------------------
# Pytest Fixtures
# --------------------------------------------------------------------
@pytest.fixture
def basic_converter():
    """
    A TypeConverter instance with just the 'basic' direct converters
    (str->int, int->str, dict->str, str->float).
    """
    return TypeConverter([
        convert_str_to_int,
        convert_int_to_str,
        convert_dict_to_str,
        convert_str_to_float,
    ])


@pytest.fixture
def parent_converter():
    """A parent converter that can convert a string to a bool."""

    def convert_str_to_bool(s: str) -> bool:
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        raise ConvertException("Cannot convert string to bool")

    return TypeConverter([convert_str_to_bool])


@pytest.fixture
def child_converter(parent_converter):
    """
    A child converter that doesn't know how to convert string->bool,
    thus falls back on the parent.
    """
    return TypeConverter([convert_str_to_int], parent=parent_converter)


@pytest.fixture
def inheritance_converter():
    """
    A TypeConverter that includes converters for:
      - dict->str, str->int, int->str, str->float (from basic)
      - base->str, str->derived
    This allows for the multi-hop chain and tests with inheritance.
    """
    return TypeConverter([
        convert_dict_to_str,
        convert_str_to_int,
        convert_int_to_str,
        convert_str_to_float,
        convert_base_to_str,
        convert_str_to_derived,
    ])


def test_direct_conversion_basic(basic_converter):
    """Test direct conversion str->int."""
    result = basic_converter.convert("123", int)
    assert result == 123
    assert isinstance(result, int)


def test_already_correct_type(basic_converter):
    """If data is already of target type, return unchanged."""
    original_value = 999
    result = basic_converter.convert(original_value, int)
    assert result is original_value  # Same object reference


def test_indirect_conversion_dict_to_float(basic_converter):
    """
    Indirect (chained) conversion: dict->str->float.
    """
    data = {"value": "123.456"}
    converted = basic_converter.convert(data, float)
    assert converted == 123.456
    assert isinstance(converted, float)


def test_parent_fallback(child_converter):
    """Child lacks str->bool, so it falls back on parent's converter."""
    result = child_converter.convert("TRUE", bool)
    assert result is True


def test_no_converter_found(basic_converter):
    """A ValueError is raised if no conversion path is found."""
    with pytest.raises(ValueError):
        basic_converter.convert(123.456, dict)  # No path to dict


def test_convert_exception_handled(basic_converter):
    """
    If a converter raises ConvertException, eventually we get ValueError
    if no alternative route is found.
    """
    with pytest.raises(ValueError):
        basic_converter.convert("not-a-number", int)


def test_text_io_wrapper_to_str_global():
    """
    Test the globally registered converter (TextIOWrapper->str).
    Use BytesIO since TextIOWrapper wraps binary streams.
    """
    pseudo_file = BytesIO(b"Hello World")
    text_wrapper = TextIOWrapper(pseudo_file, encoding="utf-8")
    result = GlobalTypeConverter.convert(text_wrapper, str)
    assert result == "Hello World"
    assert isinstance(result, str)


def test_inheritance_derived_to_str(inheritance_converter):
    """
    Derived -> str
    Should work because Derived is a subclass of Base,
    and we have a converter Base->str.
    The converter should short-circuit by noticing
    "isinstance(Derived(), Base)".
    """
    d = Derived()
    result = inheritance_converter.convert(d, str)
    # We expect the Base->str converter to run, returning the repr(d).
    assert result == repr(d)


def test_inheritance_base_to_str(inheritance_converter):
    """
    Base -> str
    Directly uses base->str.
    """
    b = Base()
    result = inheritance_converter.convert(b, str)
    assert result == repr(b)


def test_inheritance_str_to_derived(inheritance_converter):
    """
    str -> Derived
    We have a direct converter str->Derived.
    """
    result = inheritance_converter.convert("Hello", Derived)
    assert isinstance(result, Derived)
    assert result.name == "Derived from 'Hello'"


def test_inheritance_derived_to_base(inheritance_converter):
    """
    Derived -> Base
    Should short-circuit (no actual conversion needed) because
    'Derived' *is* an instance of 'Base'. We expect the same object back.
    """
    d = Derived()
    result = inheritance_converter.convert(d, Base)
    assert result is d  # same object, no conversion needed


def test_inheritance_base_to_derived_possible(inheritance_converter):
    """
    Base -> Derived
    If we define a chain:
      Base->str (via base_to_str)
      str->Derived (via str_to_derived)
    then we DO have a path.
    So this test should succeed, giving a Derived object
    whose name includes the original base's repr.
    If your domain logic says it "shouldn't exist," remove or skip this test.
    """
    b = Base(name="MyBase")
    result = inheritance_converter.convert(b, Derived)
    assert isinstance(result, Derived)
    # The derived was constructed from the string version of b
    assert "MyBase" in result.name


def test_three_hop_chain(inheritance_converter):
    """
    Test for 3 or more hops:
    dict -> str -> int -> float
    Using:
       convert_dict_to_str,
       convert_str_to_int,
       convert_int_to_str,
       convert_str_to_float
    We'll do 4 conversions in total:
      1) dict->str
      2) str->int
      3) int->str
      4) str->float
    (That's 3 "hops" in between, i.e. 4 edges.)
    """
    data = {"value": "1234"}
    # The final target is float
    result = inheritance_converter.convert(data, float)
    assert result == float(1234)
    assert isinstance(result, float)
