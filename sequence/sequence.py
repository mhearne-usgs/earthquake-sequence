# stdlib imports
from datetime import datetime, timedelta
import sqlite3
import os.path
import re
from collections import OrderedDict

# local imports
from .rates import get_data_dir, get_rates, get_day_counts, GDICT

# neic imports
from libcomcat.dataframes import get_summary_data_frame
from libcomcat.search import search
from mapio.grid2d import Grid2D
from impactutils.mapping.city import Cities
from impactutils.extern.openquake.geodetic import geodetic_distance
from impactutils.mapping.mercatormap import MercatorMap
from impactutils.mapping.scalebar import draw_scale

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

DBFILE = 'sequence.db'
EQDIST = 12000  # maximum distance in meters between two samples
EQDIST2 = 25  # maximum distance between two sequence centers
TIMEFMT = '%Y-%m-%dT%H:%M:%S'
MINEQ = 1
CITY_SEARCH_RADIUS = 4000

GRATICULE_ZORDER = 1200
SCALE_ZORDER = 1500

NUMDAYS = 3  # minimum number of days required to declare a sequence
NUMEQ = 25  # minimum number of events required to declare a sequence
DAYGAP = 3  # how many days without earthquakes before we consider end of sequence

eqtable = {'id': 'integer primary key',
           'sid': 'int',
           'code': 'text',
           'time': 'datetime',
           'latitude': 'float',
           'longitude': 'float',
           'depth': 'float',
           'magnitude': 'float'}
sqtable = {'id': 'integer primary key',
           'name': 'text',
           'start_time': 'datetime',
           'end_time': 'datetime',
           'center_lat': 'float',
           'center_lon': 'float',
           'nearest_city': 'text',
           'dist_nearest_city': 'float',
           'sequence_ended': 'boolean DEFAULT 0',
           'n_earthquakes': 'integer',
           'projstr': 'text',
           'radius': 'float',
           'xmin': 'float',
           'xmax': 'float',
           'ymin': 'float',
           'ymax': 'float'}
TABLES = {'earthquake': eqtable,
          'sequence': sqtable}

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
        self._meangrid, self._stdgrid = get_rates()
        self._dbfile = os.path.join(get_data_dir(), DBFILE)
        self._cities = Cities.fromDefault()
        if not os.path.isfile(self._dbfile):
            self._db = sqlite3.connect(self._dbfile)
            self._db.row_factory = sqlite3.Row
            self._cursor = self._db.cursor()
            create_tables(self._db, self._cursor)
        else:
            self._db = sqlite3.connect(self._dbfile)
            self._db.row_factory = sqlite3.Row
            self._cursor = self._db.cursor()

        self._debug_plot_counter = 1

    def updateSequences(self, stime):
        etime = stime + timedelta(days=1)
        events = search(starttime=stime,
                        endtime=etime,
                        minlatitude=-90,
                        maxlatitude=90,
                        minlongitude=-180,
                        maxlongitude=180,
                        minmagnitude=0.0,
                        maxmagnitude=9.9)
        todayframe = get_summary_data_frame(events)
        todaydata = get_day_counts(GDICT, todayframe)
        todaygrid = Grid2D(data=todaydata, geodict=GDICT)
        for row in range(0, GDICT.ny):
            for col in range(0, GDICT.nx):
                if row == 19 and col == 29:
                    foo = 1
                clat, clon = GDICT.getLatLon(row, col)
                tvalue = todaygrid._data[row, col]
                mvalue = self._meangrid._data[row, col]
                svalue = self._stdgrid._data[row, col]
                # thresh = tvalue > mvalue + svalue * 3
                thresh = tvalue > MINEQ
                xmin = clon - GDICT.dx / 2
                xmax = clon + GDICT.dx / 2
                ymin = clat - GDICT.dy / 2
                ymax = clat + GDICT.dy / 2
                if thresh:
                    c1 = todayframe['latitude'] > ymin
                    c2 = todayframe['latitude'] <= ymax
                    c3 = todayframe['longitude'] > xmin
                    c4 = todayframe['longitude'] <= xmax
                    cluster = todayframe[c1 & c2 & c3 & c4].copy()
                    class_frame, pproj = self.get_clusters(cluster, clon, clat)
                    self.insertSequences(class_frame, pproj)
        # call a method that filters out clusters that don't match the definition
        # of an earthquake sequence.
        self.cleanSequences()

    def cleanSequences(self):
        # deletesequences that don't match the rules
        # rules are:
        # 1) sequences that have not had an earthquake in DAYGAP days are done.
        # 2) of those, sequences must have more than NUMEQ earthquakes in them
        # 3) and those sequences must be more than NUMDAYS long

        query = 'SELECT id, start_time, end_time, n_earthquakes FROM sequence'
        dataframe = pd.read_sql_query(query, self._db, parse_dates=[
            'start_time', 'end_time'])
        dataframe['elapsed'] = dataframe['end_time'] - dataframe['start_time']

        # get the most recent update time by looking at most EQ loaded
        query2 = 'SELECT max(time) FROM earthquake'
        self._cursor.execute(query2)
        maxtime = datetime.strptime(self._cursor.fetchone()[0], TIMEFMT)

        # mark sequences that have stopped
        dataframe['gap_days'] = maxtime - dataframe['end_time']
        daygap = timedelta(days=DAYGAP)
        ended = dataframe['gap_days'] > daygap
        dataframe['ended'] = ended

        # only consider those sequences that have stopped
        dataframe = dataframe[dataframe['ended'] == True]

        # if we don't have any stopped sequences, bail out
        if not len(dataframe):
            return

        # delete small/short sequences
        c1 = dataframe['n_earthquakes'] < NUMEQ
        c2 = dataframe['elapsed'] < timedelta(days=NUMDAYS)
        deletes = dataframe[c1 | c2]
        c3 = dataframe['n_earthquakes'] >= NUMEQ
        c4 = dataframe['elapsed'] >= timedelta(days=NUMDAYS)
        keeps = dataframe[c3 & c4]

        # clean out non sequences
        for idx, row in deletes.iterrows():
            edelete = 'DELETE FROM earthquake WHERE sid=%i' % row['id']
            self._cursor.execute(edelete)
            self._db.commit()
            sdelete = 'DELETE FROM sequence WHERE id=%i' % row['id']
            self._cursor.execute(sdelete)
            self._db.commit()

        # mark dead sequences
        for idx, row in keeps.iterrows():
            smark = 'UPDATE sequence SET sequence_ended=1 WHERE id=%i' % row['id']
            self._cursor.execute(smark)
            self._db.commit()

    def insertSequences(self, class_frame, proj):
        nquery = 'SELECT count(*) FROM sequence'
        nseq = self._cursor.execute(nquery).fetchone()[0]
        classes = class_frame['class'].unique().tolist()
        if -1 in classes:
            classes.remove(-1)
        if not nseq:
            for cclass in classes:
                self.insertClass(cclass, class_frame, proj)
        else:
            for cclass in classes:
                sqstats = self.getSequenceStats(class_frame, proj)
                if sqstats['name'].startswith('West'):
                    foo = 1
                seqid = self.checkSequence(cclass, class_frame, proj)
                if seqid is None:
                    self.insertClass(cclass, class_frame, proj)
                else:
                    self.mergeSequence(seqid, cclass, class_frame, proj)

    def checkSequence(self, cclass, class_frame, proj):
        classrows = class_frame[class_frame['class'] == cclass]
        sqstats = self.getSequenceStats(classrows, proj)
        cname = sqstats['name']
        clat = sqstats['center_lat']
        clon = sqstats['center_lon']
        radius = sqstats['radius']
        seq_query = 'SELECT id, name, center_lat, center_lon, radius FROM sequence WHERE sequence_ended = 0'
        self._cursor.execute(seq_query)
        seq_rows = self._cursor.fetchall()
        if not len(seq_rows):
            return None
        for row in seq_rows:
            sid = row['id']
            name = row['name']
            center_lat = row['center_lat']
            center_lon = row['center_lon']
            sradius = row['radius']
            # what is the distance in km from one center to another?
            dist = geodetic_distance(clat, clon, center_lat, center_lon)
            if name[0:7] == cname[0:7] and name.startswith('West'):
                foo = 1
            cmp_dist = max(radius + sradius, EQDIST2)
            if dist < cmp_dist:
                return sid

        return None

    def mergeSequence(self, seqid, cclass, class_frame, proj):
        classrows = class_frame[class_frame['class'] == cclass]
        # for debugging, get the sequence stats
        query1 = 'SELECT name, center_lat, center_lon, radius, n_earthquakes FROM sequence WHERE id=%i' % seqid
        self._cursor.execute(query1)
        trow1 = self._cursor.fetchone()
        for idx, row in classrows.iterrows():
            code = row['id']
            equery = 'SELECT id FROM earthquake WHERE code="%s"' % code
            self._cursor.execute(equery)
            erow = self._cursor.fetchone()
            etime = row['time'].strftime(TIMEFMT)
            elat = row['latitude']
            elon = row['longitude']
            edep = row['depth']
            emag = row['magnitude']
            if erow is not None:
                eid = erow[0]
                cols = ['time="%s"' % etime,
                        'latitude = %.4f' % elat,
                        'longitude = %.4f' % elon,
                        'depth = %.1f' % edep,
                        'magnitude = %.1f' % emag]
                colstr = ','.join(cols)
                squery = 'UPDATE earthquake SET %s WHERE id=%i' % (colstr, eid)
            else:
                colstr = '(sid, code, time, latitude, longitude, depth, magnitude)'
                valstr = '(%i, "%s", "%s", %.4f, %.4f, %.1f, %.1f)'
                vals = valstr % (seqid, code, etime, elat, elon, edep, emag
                                 )
                tpl = (colstr, vals)
                squery = 'INSERT INTO earthquake %s VALUES %s' % tpl
            self._cursor.execute(squery)
            self._db.commit()
        # redo the sequence metadata
        query2 = 'SELECT code, time, latitude, longitude, depth, magnitude FROM earthquake WHERE sid=%i' % seqid
        df = pd.read_sql_query(query2, self._db, parse_dates=['time'])
        df = df.rename(columns={'code': 'id'})
        sqstats = self.getSequenceStats(df, proj)

        self.updateSequence(seqid, sqstats)

    def updateSequence(self, sid, sqstats):
        nuggets = ['name = "%s"' % sqstats['name'],
                   'start_time = "%s"' % sqstats['start_time'],
                   'end_time = "%s"' % sqstats['end_time'],
                   'center_lat = %.4f' % sqstats['center_lat'],
                   'center_lon = %.4f' % sqstats['center_lon'],
                   'nearest_city = "%s"' % sqstats['nearest_city'],
                   'dist_nearest_city = %.2f' % sqstats['dist_nearest_city'],
                   'n_earthquakes = %i' % sqstats['n_earthquakes'],
                   'radius = %.2f' % sqstats['radius'],
                   'xmin = %.4f' % sqstats['xmin'],
                   'xmax = %.4f' % sqstats['xmax'],
                   'ymin = %.4f' % sqstats['ymin'],
                   'ymax = %.4f' % sqstats['ymax'],
                   ]
        colstr = ','.join(nuggets)
        query = 'UPDATE sequence SET %s WHERE id=%i' % (colstr, sid)
        self._cursor.execute(query)
        self._db.commit()

    def insertSequence(self, sqstats):
        cols = ['name', 'start_time', 'end_time',
                'center_lat', 'center_lon', 'nearest_city',
                'dist_nearest_city',
                'n_earthquakes', 'radius', 'xmin', 'xmax',
                'ymin', 'ymax'
                ]
        colstr = ','.join(cols)
        fmt = '"%s", "%s", "%s", %.4f, %.4f, "%s", %.2f, %i, %.1f, %.4f, %.4f, %.4f, %.4f'
        # name = self._getMatchingName(sqstats['name'])
        if np.isnan(sqstats['dist_nearest_city']):
            distance = None
        else:
            distance = sqstats['dist_nearest_city']
        tpl = (sqstats['name'],
               sqstats['start_time'].strftime(TIMEFMT),
               sqstats['end_time'].strftime(TIMEFMT),
               sqstats['center_lat'],
               sqstats['center_lon'],
               sqstats['nearest_city'],
               sqstats['dist_nearest_city'],
               sqstats['n_earthquakes'],
               sqstats['radius'],
               sqstats['xmin'],
               sqstats['xmax'],
               sqstats['ymin'],
               sqstats['ymax'],
               )
        valstr = fmt % tpl
        query = 'INSERT INTO sequence (%s) VALUES (%s)' % (colstr, valstr)
        self._cursor.execute(query)
        self._db.commit()

    def insertClass(self, cclass, class_frame, proj):
        classrows = class_frame[class_frame['class'] == cclass]
        sqstats = self.getSequenceStats(classrows, proj)
        self.insertSequence(sqstats)
        seqid = self._cursor.execute(
            'SELECT last_insert_rowid()').fetchone()[0]
        for idx, row in classrows.iterrows():
            colstr = '(sid, code, time, latitude, longitude, depth, magnitude)'
            valstr = '(%i, "%s", "%s", %.4f, %.4f, %.1f, %.1f)'
            tpl = (seqid, row['id'], row['time'],
                   row['latitude'], row['longitude'],
                   row['depth'], row['magnitude'])
            query2 = 'INSERT INTO earthquake %s VALUES %s' % (
                colstr, valstr % tpl)
            self._cursor.execute(query2)
            self._db.commit()

    def getSequenceStats(self, class_frame, proj):
        start_time = class_frame['time'].min()
        end_time = class_frame['time'].max()
        class_lat = class_frame['latitude'].tolist()
        class_lon = class_frame['longitude'].tolist()
        x, y = proj(class_lon, class_lat)
        x = np.array(x)
        y = np.array(y)
        cx = np.mean(x)
        cy = np.mean(y)
        clon, clat = proj(cx, cy, inverse=True)
        near_cities = self._cities.limitByRadius(
            clat, clon, CITY_SEARCH_RADIUS).getDataFrame()
        citylat = near_cities['lat']
        citylon = near_cities['lon']
        dist = geodetic_distance(clon, clat, citylon, citylat)
        near_cities['distance'] = dist
        near_cities = near_cities.sort_values('distance')

        # get the distance from the center of mass to all the points
        dxsq = np.power((x - cx), 2)
        dysq = np.power((y - cy), 2)
        cdist = np.sqrt(dxsq + dysq)
        radius = cdist.max() / 1000
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        ulx, uly = proj(xmin, ymax, inverse=True)
        lrx, lry = proj(xmax, ymin, inverse=True)
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
                    'xmin': ulx,
                    'xmax': lrx,
                    'ymin': lry,
                    'ymax': uly}
        return sequence

    def _getMatchingName(self, name):
        query = 'SELECT name FROM sequence WHERE name LIKE "%s%%"' % name
        self._cursor.execute(query)
        rows = self._cursor.fetchall()
        if not len(rows):
            return name
        oldname = None
        for row in rows:
            if not row['name'].startswith(name):
                continue
            if oldname is None:
                oldname = row['name']
                continue
            if len(row['name']) > len(oldname):
                oldname = row['name']
        if oldname is None:
            return name

        parts = oldname.split('_')
        if len(parts) == 1:
            version = 1
            newname = '%s_%i' % (name, version)
            return newname
        try:
            version = int(parts[-1])
            newname = '%s_%i' % (name, version + 1)
            return newname
        except:
            return name

    def get_clusters(self, dataframe, clon, clat):
        plat = dataframe['latitude'].values
        plon = dataframe['longitude'].values
        paramstr = '+proj=merc +lat_ts=%.4f +lon_0=%.4f'
        params = paramstr % (clat, clon)
        proj = Proj(params)
        x, y = proj(plon, plat)
        X = np.array(list(zip(x, y)))
        clusterer = DBSCAN(eps=EQDIST, min_samples=3)
        clusterer = clusterer.fit(X)
        dataframe['class'] = clusterer.labels_.copy()  # -1 is noise

        # try to expand each cluster, by calculating the center point
        # and the minimum bounding radius, then grab all points
        # inside that radius.
        # classes = dataframe['class'].unique().tolist()
        # if -1 in classes:
        #     classes.remove(-1)

        # # get the noise points
        # c2 = dataframe['class'] == -1
        # nx = x[c2]
        # ny = y[c2]
        # for cclass in classes:
        #     c1 = dataframe['class'] == cclass
        #     cx = x[c1]
        #     cy = y[c1]
        #     mx = np.mean(cx)
        #     my = np.mean(cy)
        #     dx = cx - mx
        #     dy = cy - my
        #     center_dist = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        #     radius = center_dist.max()
        #     # now get the distance to any noise points from the center
        #     dx2 = nx - mx
        #     dy2 = ny - my
        #     noise_dist = np.sqrt(np.power(dx2, 2) + np.power(dy2, 2))
        #     c3 = noise_dist < radius
        #     if c3.any():
        #         foo = 1

        # since DBSCAN doesn't always give us all of the points
        # we want, we'll use pdist from scipy to get all distances
        # between our points.
        # dist = pdist(X, 'euclidean') / 1000

        # find the distances associated with all unclassified points
        # idx = dist < EQDIST / 1000
        # nearidx = np.where(idx)[0]
        # if (dataframe['class'] > -1).any():
        #     self._debug_plot(dataframe, proj)
        # for nidx in nearidx:
        #     idx1, idx2 = get_pair_idx(nidx, len(dist))
        #     classval1 = dataframe['class'].iloc[idx1]
        #     classval2 = dataframe['class'].iloc[idx2]
        #     foo = 1

        return (dataframe, proj)

    def getSequences(self, confirmed=False):
        #            'name': 'text',
        #    'start_time': 'datetime',
        #    'end_time': 'datetime',
        #    'center_lat': 'float',
        #    'center_lon': 'float',
        #    'radius': 'float',
        #    'xmin': 'float',
        #    'xmax': 'float',
        #    'ymin': 'float',
        #    'ymax': 'float'}
        columns = ['name', 'start_time', 'end_time',
                   'center_lat', 'center_lon', 'nearest_city',
                   'xmin', 'xmax', 'ymin', 'ymax',
                   'radius', 'n_earthquakes', 'sequence_ended']
        seqlist = []
        query = 'SELECT * FROM sequence'
        parse_dates = [
            'start_time', 'end_time']
        dataframe = pd.read_sql_query(query, self._db, parse_dates=parse_dates)
        dataframe = dataframe[columns]
        if confirmed:
            dataframe['elapsed'] = dataframe['end_time'] - \
                dataframe['start_time']
            c1 = dataframe['n_earthquakes'] >= NUMEQ
            c2 = dataframe['elapsed'] >= timedelta(days=NUMDAYS)
            dataframe = dataframe[c1 & c2]
        return dataframe

    def getSequenceEvents(self, name):
        query = 'SELECT id FROM sequence WHERE name="%s"' % name
        self._cursor.execute(query)
        row = self._cursor.fetchone()
        if row is None:
            return None
        sid = row['id']
        equery = 'SELECT code, time, latitude, longitude, depth, magnitude FROM earthquake WHERE sid=%i' % sid
        event_table = pd.read_sql_query(equery, self._db, parse_dates=['time'])
        event_table = event_table.rename(columns={'code': 'id'})
        return event_table

    def getSequence(self, seqname):
        columns = ['name', 'start_time', 'end_time',
                   'center_lat', 'center_lon', 'nearest_city',
                   'xmin', 'xmax', 'ymin', 'ymax',
                   'radius', 'n_earthquakes', 'sequence_ended']
        seqlist = []
        query = 'SELECT * FROM sequence WHERE name="%s"'
        parse_dates = [
            'start_time', 'end_time']
        dataframe = pd.read_sql_query(query, self._db, parse_dates=parse_dates)
        dataframe = dataframe[columns]
        sequence = dataframe.iloc[0].to_dict()
        return sequence

    def mapSequence(self, sequence):
        sequence_frame = self.getSequenceEvents(sequence['name'])
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
        return ax, fig

    def plotSequence(self, sequence):
        df = self.getSequenceEvents(sequence['name'])
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
        return ax


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


def get_pair_idx(idx, N):
    npairs = (N * (N - 1)) / 2
    i1 = []
    i2 = []
    ni1 = N - 1
    for i in range(0, N):
        i1t = [i] * ni1
        i2start = i + 1
        i2end = i2start + ni1
        i2t = np.arange(i2start, i2end, dtype=np.int32)
        i1.append(i1t)
        i2.append(i2t)
    idx1 = i1[idx]
    idx2 = i2[idx]
    return (idx1, idx2)
