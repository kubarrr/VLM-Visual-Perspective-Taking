import ast


def llm_output_to_list(output: str):
    """
    Safely decode a string representing a list (from LLM output) to a Python list.
    """
    result = ast.literal_eval(output)
    if isinstance(result, list):
        return result
    return None

 
