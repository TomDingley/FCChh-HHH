"""Write explicit TTrees for compatibility with the analysis ROOT release."""
import awkward as ak
import uproot


def open_root(path):
    """Read local ROOT files directly, without an asynchronous filesystem backend."""
    return uproot.open(path, handler=uproot.source.file.MemmapSource)


def write_events(path, arrays):
    columns = {name: ak.Array(values) for name, values in arrays.items()}
    if not columns:
        raise ValueError('Cannot write an events TTree without columns.')
    if len({len(values) for values in columns.values()}) != 1:
        raise ValueError('Event columns have different lengths.')
    with uproot.recreate(path) as output:
        tree = output.mktree('events', {name: ak.type(values).content
                                      for name, values in columns.items()})
        if len(next(iter(columns.values()))):
            tree.extend(columns)
