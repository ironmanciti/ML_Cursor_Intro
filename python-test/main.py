"""Example script demonstrating how to load and use LangChain math tools."""

from langchain_math_tools import get_math_tools


def main():
    tools = get_math_tools()

    # Each tool is a LangChain tool wrapped function; call .run to execute.
    add_tool = next(tool for tool in tools if tool.name == "add")
    multiply_tool = next(tool for tool in tools if tool.name == "multiply")

    result_add = add_tool.invoke({"a": 3, "b": 7})
    result_multiply = multiply_tool.invoke({"a": 4, "b": 5})

    print("Addition result (3 + 7):", result_add)
    print("Multiplication result (4 * 5):", result_multiply)


if __name__ == "__main__":
    main()
