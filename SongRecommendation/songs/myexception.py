class InvalidInputError(Exception):
    """
    This InvalidInputError class extends from the Extension class.
    """
    def __str__(self):
        return 'This is an invalid input. Please enter either value 0 or 1.'

