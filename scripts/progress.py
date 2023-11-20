from tqdm import tqdm



class ShowProgress(tqdm):
    """
    Class used to wrap tqdm into a more readable iterator.
    Handles a bunch of default arguments so that they won't need to be repeated.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("desc", "")
        kwargs.setdefault("position", 0)
        kwargs.setdefault("ncols", 100)
        kwargs.setdefault("leave", True)
        kwargs.setdefault("ascii", False)
        super().__init__(*args, **kwargs)