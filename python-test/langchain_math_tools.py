"""Utility tools for basic arithmetic operations compatible with LangChain agents."""

from typing import Union

from langchain.agents import tool

Number = Union[int, float]


@tool
def add(a: Number, b: Number) -> Number:
    """Add two numbers together and return the result."""
    return a + b


@tool
def multiply(a: Number, b: Number) -> Number:
    """Multiply two numbers together and return the product."""
    return a * b


def get_math_tools():
    """Return the LangChain tools that provide addition and multiplication."""
    return [add, multiply]
