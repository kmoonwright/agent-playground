"""Where tracks come from and where they are kept.

`source.py` holds the protocol every source implements; adding a source means one
new module here plus a name and a constructor branch in `registry.py`.
`library.py` is the on-disk cache those sources feed.
"""
