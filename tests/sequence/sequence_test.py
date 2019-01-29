#!/usr/bin/env python
import os.path
from collections import OrderedDict
from datetime import datetime, timedelta

import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import logging
import logging.config

import matplotlib.patches as mpatches
from impactutils.mapping.city import Cities

import pyproj
import numpy as np

from sequence.sequence2 import SequenceDetector
from sequence.seqdb import merge_sequences, boxes_intersect


def get_logger():
    level = logging.DEBUG
    fmt = '%(levelname)s -- %(asctime)s -- %(module)s.%(funcName)s -- %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logdict = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': fmt,
                'datefmt': datefmt
            }
        },
        'handlers': {
            'stream': {
                'level': level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler'
            }
        },
        'loggers': {
            '': {
                'handlers': ['stream'],
                'level': level,
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(logdict)
    # Get the root logger, otherwise we can't log in sub-libraries
    logger = logging.getLogger()
    return logger


def merge_test():
    id1 = [1, 2, 4, 5, 7, 1]
    id2 = [2, 3, 5, 6, 8, 9]
    sets = merge_sequences(id1, id2)
    set1 = set([1, 2, 3, 9])
    set2 = set([4, 5, 6])
    set3 = set([7, 8])
    assert sets[0] == set1
    assert sets[1] == set2
    assert sets[2] == set3


def intersect_test():
    bounds1 = (-116.80, -116.78, 33.49, 33.50)
    bounds2 = (-155.48, -110.75, 19.18, 63.10)
    boxes_intersect(bounds1, bounds2)


def detector_test():
    logger = get_logger()
    detector = SequenceDetector()
    # tdate1 = datetime(2017, 9, 2)
    # tdate2 = datetime(2017, 9, 3)
    # detector.updateSequences(tdate1)
    # seqinfo = detector.getSequences()
    # detector.updateSequences(tdate2)
    # seqlist = detector.getSequences()

    # danville CA sequence

    stime = datetime(2018, 1, 1)
    etime = datetime(2018, 2, 28)
    # etime = datetime(2018, 1, 4)
    while stime < etime:
        print('Updating %s...' % (str(stime)))
        detector.updateSequences(stime)
        stime += timedelta(days=1)

    sequence_names = detector.getSequenceNames(confirmed=True)

    for name in sequence_names:
        sequence, seqframe = detector.getSequence(name)

        # map sequence
        ax, fig = detector.mapSequence(sequence, seqframe)
        mapfile = '%s_map.png' % sequence['name'].replace(' ', '_')
        mapfile = mapfile.replace('-', '_')
        mapfile = os.path.join(os.path.expanduser('~'), mapfile)
        plt.savefig(mapfile)
        plt.close('all')

        # plot time vs mag
        ax = detector.plotSequence(sequence, seqframe)
        pltfile = '%s_plot.png' % sequence['name'].replace(' ', '_')
        pltfile = pltfile.replace('-', '_')
        pltfile = os.path.join(os.path.expanduser('~'), pltfile)
        plt.savefig(pltfile)
        plt.close('all')


if __name__ == '__main__':
    intersect_test()
    merge_test()
    detector_test()
