#!/usr/bin/env python
"""
@name: evaluation.py
@desc: Utility functions for evaluating msuic structure segmentation
       performance.
@auth: Tim O'Brien
@date: Feb. 16, 2016
"""

import mir_eval
import os
import datapaths as dpath

def load_salami(filename):
    """
    Load SALAMI pre-parsed annotations.
    Input   : filename (of SALAMI annotation)
    Outputs : intervals, labels

    Credit: craffel, https://craffel.github.io/mir_eval/
    """
    events = mir_eval.io.load_labeled_events(filename)
    intervals = mir_eval.util.boundaries_to_intervals(events[0])
    labels = events[1][:-1]
    return intervals, labels

def id2filenames(sid, ann_type="uppercase", salamipath=dpath.SALAMI):
    """
    Returns filenames for SALAMI annotation text files, based on the SALAMI
    ID and the type (e.g. uppercase, lowercase, function).
    """
    import fnmatch

    spath = os.path.join(
        salamipath,
        "annotations",
        str(sid),
        "parsed"
        )
    files = None
    if os.path.exists(spath):
        files = os.listdir(spath)
        files = fnmatch.filter(files, '*'+ann_type+'*')
        # Prepend directory path
        for i in range(len(files)):
            files[i] = os.path.join(spath, files[i])
    return files

def id2annotations(sid, ann_type="uppercase", salamipath=dpath.SALAMI):
    """
    Given a SALAMI id number, return the annotations in a dictionary.
    """
    files = id2filenames(sid, ann_type=ann_type, salamipath=salamipath)
    annotations = {}
    for i in range(len(files)):
        annotations[os.path.basename(files[i])] = load_salami(files[i])
    return annotations

def id2segtimes(sid, ann_type="uppercase", salamipath=dpath.SALAMI):
    """
    Given a SALAMI id number, return the annotated segmentation
    times.
    """
    files = id2filenames(sid, ann_type=ann_type, salamipath=salamipath)
    times = []
    for i in range(len(files)):
        events, _ = mir_eval.io.load_labeled_events(files[i])
        times = times + events[1:-1].tolist()
    return times

if __name__ == "__main__":
    pass

