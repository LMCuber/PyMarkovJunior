from pprint import pprint


def is_true(string):
    return string.casefold() == "true"  # casefold is a more rigorous .lower() I heard somewhere
