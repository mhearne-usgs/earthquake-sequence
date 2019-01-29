# stdlib imports
import os.path
from datetime import datetime, timedelta
import sys

# third party imports
import numpy as np
import pandas as pd

# neic imports
from mapio.grid2d import Grid2D
from mapio.geodict import GeoDict
from mapio.writer import write
from mapio.reader import read
from libcomcat.search import search
from libcomcat.dataframes import get_summary_data_frame

STDFILE = 'stdgrid.cdf'
MEANFILE = 'meangrid.cdf'
FRAMEFILE = 'allquakes.csv'
GDICT = GeoDict({'xmin': -180, 'xmax': 180,
                 'ymin': -76, 'ymax': 76,
                 'dx': 2, 'dy': 2,
                 'nx': 181, 'ny': 77})


def get_day_counts(gdict, dayrows):
    today = np.zeros((gdict.ny, gdict.nx))
    for idx, drow in dayrows.iterrows():
        lat = drow['latitude']
        lon = drow['longitude']
        if lat < gdict.ymin or lat > gdict.ymax:
            continue
        row, col = gdict.getRowCol(lat, lon)
        today[row, col] += 1
    return today


def get_data_dir():
    seqdir = os.path.join(os.path.expanduser('~'), '.sequence')
    if not os.path.isdir(seqdir):
        os.mkdir(seqdir)
    return seqdir


def get_rates():
    data_dir = get_data_dir()
    meanfile = os.path.join(data_dir, MEANFILE)
    stdfile = os.path.join(data_dir, STDFILE)
    framefile = os.path.join(data_dir, FRAMEFILE)
    if not os.path.isfile(framefile):
        ndays = 365
        interval = 1
        today = datetime.utcnow()
        stime = today - timedelta(days=ndays)
        etime = stime + timedelta(days=interval)
        dataframe = None
        while etime < today:
            print(stime)
            try:
                events = search(starttime=stime,
                                endtime=etime,
                                minlatitude=-90,
                                maxlatitude=90,
                                minlongitude=-180,
                                maxlongitude=180,
                                minmagnitude=0.0,
                                maxmagnitude=9.9)
            except Exception as e:
                try:
                    events = search(starttime=stime,
                                    endtime=etime,
                                    minlatitude=-90,
                                    maxlatitude=90,
                                    minlongitude=-180,
                                    maxlongitude=180,
                                    minmagnitude=0.0,
                                    maxmagnitude=9.9)
                except Exception as e:
                    print('Bah humbug.')
                    sys.exit(1)
            if dataframe is None:
                dataframe = get_summary_data_frame(events)
            else:
                df = get_summary_data_frame(events)
                dataframe = pd.concat([dataframe, df])
            stime = etime
            etime = stime + timedelta(days=interval)
        dataframe.to_csv(framefile)
    else:
        dataframe = pd.read_csv(framefile, parse_dates=['time'])
        # get all of our desired statistics
        if not os.path.isfile(meanfile):
            meangrid, stdgrid = assign_events(GDICT, dataframe)
            write(meangrid, meanfile, 'netcdf')
            write(stdgrid, stdfile, 'netcdf')
        else:
            meangrid = read(meanfile)
            stdgrid = read(stdfile)

    return (meangrid, stdgrid)


def assign_events(gdict, dataframe):
    dt = dataframe['time'].max() - dataframe['time'].min()
    ndays = int(np.floor(dt.total_seconds() / SECS_PER_DAY))
    countdata = np.zeros((gdict.ny, gdict.nx, ndays))

    # loop over days in dataframe, and assign events to the corresponding day
    # in countdata.
    start_day = dataframe['time'].min()
    end_day = dataframe['time'].max()
    for dayidx in range(0, ndays):
        today_start = start_day + timedelta(days=dayidx)
        today_end = today_start + timedelta(days=1)
        c1 = dataframe['time'] >= today_start
        c2 = dataframe['time'] < today_end
        dayrows = dataframe[(c1) & (c2)]
        countdata[:, :, dayidx] = get_day_counts(gdict, dayrows)

    meandata = np.mean(countdata, axis=2)
    sqdata = np.zeros((gdict.ny, gdict.nx, ndays))
    for i in range(0, ndays):
        sqdata[:, :, i] = np.power(countdata[:, :, i] - meandata, 2)

    stddata = np.sqrt(np.sum(sqdata, axis=2) / (ndays - 1))
    meangrid = Grid2D(data=meandata, geodict=gdict)
    stdgrid = Grid2D(data=stddata, geodict=gdict)
    return (meangrid, stdgrid)
