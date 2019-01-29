# stdlib imports
from datetime import datetime, timedelta
import sqlite3
import os.path
import re
from collections import OrderedDict
import time
import logging

# third party imports
from pyproj import Proj
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from scipy.misc import factorial
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn.cluster import DBSCAN
import yaml

# usgs imports
from libcomcat.dataframes import get_summary_data_frame
from libcomcat.search import search
from mapio.geodict import GeoDict
from impactutils.mapping.city import Cities
from impactutils.extern.openquake.geodetic import geodetic_distance
from impactutils.mapping.mercatormap import MercatorMap
from impactutils.mapping.scalebar import draw_scale

# local imports
from .seqdb import SequenceDatabase
from .rates import get_data_dir

DBFILE = 'sequence.db'
CFGFILE = 'cfgsequence.yml'

GRATICULE_ZORDER = 1200
SCALE_ZORDER = 1500

MSIZES = OrderedDict([(0.0, 6),
                      (1.0, 8),
                      (2.0, 10),
                      (3.0, 12),
                      (4.0, 14),
                      (5.0, 16),
                      (6.0, 18),
                      (7.0, 20)])


class SequenceDetector(object):
    def __init__(self):

        dbfile = os.path.join(get_data_dir(), DBFILE)
        cfgfile = os.path.join(get_data_dir(), CFGFILE)
        self._config = yaml.load(open(cfgfile, 'rt'))
        self._cities = Cities.fromDefault()
        create = False
        if not os.path.isfile(dbfile):
            create = True
        self._seqdb = SequenceDatabase(dbfile, self._config, create=create)

    def updateSequences(self, stime):
        etime = stime + timedelta(days=1)
        t1 = time.time()
        events = search(starttime=stime,
                        endtime=etime,
                        minlatitude=-90,
                        maxlatitude=90,
                        minlongitude=-180,
                        maxlongitude=180,
                        minmagnitude=0.0,
                        maxmagnitude=9.9)
        todayframe = get_summary_data_frame(events)
        logging.info('Got day data...')
        gdict = GeoDict(self._config['GDICT'])
        for row in range(0, gdict.ny):
            for col in range(0, gdict.nx):
                clat, clon = gdict.getLatLon(row, col)
                xmin = clon - gdict.dx / 2
                xmax = clon + gdict.dx / 2
                ymin = clat - gdict.dy / 2
                ymax = clat + gdict.dy / 2

                c1 = todayframe['latitude'] > ymin
                c2 = todayframe['latitude'] <= ymax
                c3 = todayframe['longitude'] > xmin
                c4 = todayframe['longitude'] <= xmax
                gridframe = todayframe[c1 & c2 & c3 & c4].copy()
                if not len(gridframe):
                    continue
                cluster_list, pproj = self.getClusters(gridframe)
                if len(cluster_list):
                    self.insertClusters(cluster_list, pproj)

        # now we need to merge sequences that may have fallen on the edge of
        # a grid border.
        logging.info('Matching sequences...')
        sequence_sets = self._seqdb.mergeSequences()
        for tseqset in sequence_sets:
            seqset = sorted(list(tseqset))
            id1 = seqset[0]
            frame1 = self._seqdb.getSequenceEvents(id1)
            xmin1 = frame1['longitude'].min()
            xmax1 = frame1['longitude'].max()
            ymin1 = frame1['latitude'].min()
            ymax1 = frame1['latitude'].max()
            bounds1 = (xmin1, xmax1, ymin1, ymax1)
            for i in range(1, len(seqset)):
                id2 = seqset[i]
                frame2 = self._seqdb.getSequenceEvents(id2)
                xmin2 = frame2['longitude'].min()
                xmax2 = frame2['longitude'].max()
                ymin2 = frame2['latitude'].min()
                ymax2 = frame2['latitude'].max()
                bounds2 = (xmin2, xmax2, ymin2, ymax2)
                logging.info('Merging sequence %i and %i' % (id1, id2))
                fmt = 'Bounds1: %.2f, %.2f, %.2f, %.2f'
                logging.info(fmt % (bounds1))
                fmt = 'Bounds2: %.2f, %.2f, %.2f, %.2f'
                logging.info(fmt % (bounds2))
                dataframe = pd.concat([frame1, frame2], axis=0)
                if dataframe.duplicated('id').any():
                    foo = 1
                proj = self.getProj(dataframe)
                sqstats = self.getSequenceStats(dataframe, proj)
                self._seqdb.updateSequence(frame2, sqstats, id1)
                self._seqdb.deleteSequence(id2)

        logging.info('Finding stopped sequences...')
        # now find all sequences that seem to have ended (using config criteria)
        ended = self._seqdb.getStoppedDataframe()
        if len(ended) > 0:
            x = 1
        self._seqdb.markStoppedSequences(ended)

        logging.info('Finding non sequences...')
        # find all ended sequences that don't match our criteria for a sequence
        deletes = self._seqdb.getNonSequences()
        if len(deletes) > 0:
            x = 1
        self._seqdb.deleteNonSequences(deletes)
        t2 = time.time()
        # print('%s elapsed: %.1f' % (str(stime), (t2 - t1)))

    def insertClusters(self, cluster_list, proj):
        nseq = self._seqdb.getNumSequences()
        if not nseq:
            for cluster in cluster_list:
                self.insertCluster(cluster, proj)
        else:
            for cluster in cluster_list:
                merged = self.mergeCluster(cluster, proj)

    def mergeCluster(self, cluster, proj):
        sqstats = self.getSequenceStats(cluster, proj)
        seqid = self._seqdb.checkSequence(sqstats)
        merged = False
        if seqid is None:
            seqid = self._seqdb.insertSequence(cluster, sqstats)
        else:
            oldframe = self._seqdb.getSequenceEvents(seqid)
            cluster = cluster.drop(['location', 'url', 'class'], axis=1)
            oldframe = oldframe[cluster.columns]
            dataframe = pd.concat([cluster, oldframe], axis=0)
            if dataframe.duplicated('id').any():
                foo = 1
            sqstats = self.getSequenceStats(dataframe, proj)
            self._seqdb.updateSequence(cluster, sqstats, seqid)
            merged = True
        return merged

    def insertCluster(self, clusterframe, proj):
        sqstats = self.getSequenceStats(clusterframe, proj)
        seqid = self._seqdb.insertSequence(clusterframe, sqstats)
        if (sqstats['xmax'] - sqstats['xmin']) > 2:
            foo = 1
        if (sqstats['ymax'] - sqstats['ymin']) > 2:
            foo = 1

    def getClusters(self, dataframe):
        plat = dataframe['latitude'].values
        plon = dataframe['longitude'].values
        proj = self.getProj(dataframe)
        x, y = proj(plon, plat)
        X = np.array(list(zip(x, y)))
        clusterer = DBSCAN(eps=self._config['EQDIST'], min_samples=3)
        clusterer = clusterer.fit(X)
        dataframe['class'] = clusterer.labels_.copy()  # -1 is noise

        cluster_list = []
        classes = dataframe['class'].unique()
        for cclass in classes:
            if cclass == -1:
                continue
            classframe = dataframe[dataframe['class'] == cclass].copy()
            cluster_list.append(classframe)

        return (cluster_list, proj)

    def getProj(self, dataframe):
        plat = dataframe['latitude'].values
        plon = dataframe['longitude'].values
        clat = (plat.min() + plat.max()) / 2
        clon = (plon.min() + plon.max()) / 2
        paramstr = '+proj=merc +lat_ts=%.4f +lon_0=%.4f'
        params = paramstr % (clat, clon)
        proj = Proj(params)
        return proj

    def getSequenceStats(self, class_frame, proj):
        start_time = class_frame['time'].min()
        end_time = class_frame['time'].max()
        class_lat = class_frame['latitude'].tolist()
        class_lon = class_frame['longitude'].tolist()
        xmin = np.min(class_lon)
        xmax = np.max(class_lon)
        ymin = np.min(class_lat)
        ymax = np.max(class_lat)
        clon = (xmin + xmax) / 2
        clat = (ymin + ymax) / 2

        cradius = self._config['CITY_SEARCH_RADIUS']
        near_cities = self._cities.limitByRadius(
            clat, clon, cradius).getDataFrame()
        citylat = near_cities['lat']
        citylon = near_cities['lon']
        dist = geodetic_distance(clon, clat, citylon, citylat)
        near_cities['distance'] = dist
        near_cities = near_cities.sort_values('distance')

        dlon = (xmax - xmin) / 2 * 111 * np.cos(np.radians(clat))
        dlat = (ymax - ymin) / 2 * 111
        radius = np.mean([dlon, dlat])

        if radius > 200:
            foo = 1

        # Earthquake name should NearestCity_YYYY, followed by _1, _2
        # when there are multiple sequences near that city in that year
        try:
            nearest_city = near_cities.iloc[0]
            name = nearest_city['name'] + start_time.strftime('%Y%m%d%H%M%S')
            name = re.sub('\s+', '_', name)
        except IndexError:
            name = 'No_Nearest_City' + start_time.strftime('%Y%m%d%H%M%S')
            nearest_city = {'name': 'No_Nearest_City',
                            'distance': np.nan}
        sequence = {'name': name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'n_earthquakes': len(class_frame),
                    'center_lat': clat,
                    'center_lon': clon,
                    'nearest_city': nearest_city['name'],
                    'dist_nearest_city': nearest_city['distance'],
                    'radius': radius,
                    'projstr': proj.srs,
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax}
        return sequence

    def getSequenceNames(self, confirmed=False):
        return self._seqdb.getSequenceNames(confirmed=confirmed)

    def getSequence(self, seqname):
        return self._seqdb.getSequence(seqname)

    def mapSequence(self, sequence, sequence_frame):
        fig = plt.figure(figsize=[7, 7])
        clon = sequence['center_lon']
        clat = sequence['center_lat']
        xmin = sequence['xmin'] - 1
        xmax = sequence['xmax'] + 1
        ymin = sequence['ymin'] - 1
        ymax = sequence['ymax'] + 1
        bounds = [xmin, xmax, ymin, ymax]
        figsize = (7, 7)
        cities = Cities.fromDefault()
        dims = [0.1, 0.1, 0.8, 0.8]
        mmap = MercatorMap(bounds, figsize, cities, dimensions=dims)
        fig = mmap.figure
        ax = mmap.axes
        proj = mmap.proj
        markersizes = list(MSIZES.values())
        for idx, row in sequence_frame.iterrows():
            elat = row['latitude']
            elon = row['longitude']
            emag = row['magnitude']
            mdiff = np.abs(emag - np.array(list(MSIZES.keys())))
            imin = mdiff.argmin()
            markersize = markersizes[imin]
            zorder = 1 / markersize
            ax.plot([elon], [elat], 'g', marker='o', mec='k',
                    markersize=markersize, zorder=zorder,
                    transform=ccrs.PlateCarree())
        mmap.drawCities(draw_dots=True)
        _draw_graticules(ax, xmin, xmax, ymin, ymax)
        corner = 'll'
        ax.coastlines(resolution='50m')
        draw_scale(ax, corner, pady=0.05, padx=0.05, zorder=SCALE_ZORDER)

        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')
        plt.title('%s (N=%i)' % (sequence['name'], sequence['n_earthquakes']))
        return (ax, fig)

    def plotSequence(self, sequence, df):
        fig = plt.figure(figsize=(7, 7))
        mags = df['magnitude']
        times = df['time']
        mintime = times.min()
        dt = times - mintime
        secs = np.array([dti.total_seconds() for dti in dt])
        days = secs / 86400
        plt.plot(days, mags, 'b.')
        # ax = df.plot(x='time', y='magnitude', kind='scatter', figsize=(7, 7))
        plt.title('%s (N=%i)' % (sequence['name'], sequence['n_earthquakes']))
        plt.xlabel('Days Since Start of Sequence')
        plt.ylabel('Magnitude')
        ax = plt.gca()
        return (ax, fig)


def _draw_graticules(ax, xmin, xmax, ymin, ymax):
    """Draw map graticules, tick labels on map axes.
    Args:
        ax (GeoAxes): Cartopy GeoAxes.
        xmin (float): Left edge of map (degrees).
        xmax (float): Right edge of map (degrees).
        ymin (float): Bottom edge of map (degrees).
        ymax (float): Bottom edge of map (degrees).
    """
    gl = ax.gridlines(draw_labels=True,
                      linewidth=0.5, color='k',
                      alpha=0.5, linestyle='-',
                      zorder=GRATICULE_ZORDER)
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True

    # create a dictionary with the intervals we want for a span
    # of degrees.
    spans = {1: 0.25,
             2: 0.5,
             3: 1.0,
             5: 1.0,
             7: 2.0}

    span_keys = np.array(sorted(list(spans.keys())))

    nearest_xspan_idx = np.argmin(np.abs(int((xmax - xmin)) - span_keys))
    x_interval = spans[span_keys[nearest_xspan_idx]]

    nearest_yspan_idx = np.argmin(np.abs(int((ymax - ymin)) - span_keys))
    y_interval = spans[span_keys[nearest_yspan_idx]]

    # let's floor/ceil the edges to nearest 1/interval
    gxmin = x_interval * np.floor(xmin / x_interval)
    gxmax = x_interval * np.ceil(xmax / x_interval)
    gymin = y_interval * np.floor(ymin / y_interval)
    gymax = y_interval * np.ceil(ymax / y_interval)

    # check for meridian crossing
    crosses = False
    if gxmax < 0 and gxmax < gxmin:
        crosses = True
        gxmax += 360

    ylocs = np.arange(gymin, gymax + y_interval, y_interval)
    xlocs = np.arange(gxmin, gxmax + x_interval, x_interval)

    if crosses:
        xlocs[xlocs > 180] -= 360

    gl.xlocator = mticker.FixedLocator(xlocs)
    gl.ylocator = mticker.FixedLocator(ylocs)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
