def calculator(a, b, operation):
    """
    A complete calculator function that performs basic operations.

    Parameters:
        a (float): The first operand.
        b (float): The second operand.
        operation (str): The operation to perform ('add', 'subtract', 'multiply', 'divide').

    Returns:
        float: The result of the calculation.
        str: Error message if the operation is invalid or division by zero occurs.
    """
    try:
        if operation == 'add':
            return a + b
        elif operation == 'subtract':
            return a - b
        elif operation == 'multiply':
            return a * b
        elif operation == 'divide':
            if b == 0:
                return "Error: Division by zero is not allowed."
            return a / b
        else:
            return "Error: Invalid operation. Supported operations are 'add', 'subtract', 'multiply', 'divide'."
    except Exception as e:
        return f"Error: {str(e)}"