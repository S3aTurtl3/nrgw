import hashlib
def get_unique_identifier(info: dict):
    """Returns a string, short enough to be part of a filename, such that get_unique_identifier(a) == get_unique_identifier(b) if and only if a and b have the same key:value pairs"""
    unique_string = str(list(info.items()).sort())
    unique_string = hashlib.md5(unique_string).hexdigest()
    return unique_string