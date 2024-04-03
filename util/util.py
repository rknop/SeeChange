import pathlib
import collections.abc


def ensure_file_does_not_exist( filepath, delete=False ):
    """Check if a file exists.  Delete it, or raise an exception, if it does.

    Will always raise a FileExistsError if the file exists but isn't a normal file.

    Parameters
    ----------
    filepath: str or Path
       Path to the file
    delete: bool
       If True, will delete the file if it exists and is a regular
       file.  If False (default), will raise a FileExistsError
    """

    filepath = pathlib.Path( filepath )
    if filepath.exists():
        if not filepath.is_file():
            raise FileExistsError( f"{filepath} exists but is not a regular file" )
        if not delete:
            raise FileExistsError( f"{filepath} exists and delete is False" )
        else:
            filepath.unlink()


def listify( val, require_string=False ):
    """Return a list version of val.

    If val is already a sequence other than a string, return list(val).
    Otherwise, return [val].  If val is None, return None.

    Parameters
    ----------
    require_string: bool (default False)
       If true, then val must either be a sequence of strings or a string

    Returns
    -------
    list or None

    """

    if val is None:
        return val

    if isinstance( val, collections.abc.Sequence ):
        if isinstance( val, str ):
            return [ val ]
        else:
            if require_string and ( not all( [ isinstance( i, str ) for i in val ] ) ):
                raise TypeError( f'listify: all elements of passed sequence must be strings.' )
            return list( val )
    else:
        if require_string and ( not isinstance( val, str ) ):
            raise TypeError( f'listify wants a string, not a {type(val)}' )
        return [ val ]


def remove_empty_folders(path, remove_root=True):
    """Recursively remove any empty folders in the given path.

    Parameters
    ----------
    path: str or pathlib.Path
        The path to remove empty folders from.
    remove_root: bool
        If True, remove the root folder as well if it is empty.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        for subpath in path.iterdir():
            remove_empty_folders(subpath, remove_root=True)
        if remove_root and not any(path.iterdir()):
            path.rmdir()
