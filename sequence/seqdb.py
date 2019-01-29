import sqlite3
from collections import OrderedDict
from datetime import datetime, timedelta
import logging

import pandas as pd
from shapely.geometry import Polygon

from impactutils.extern.openquake.geodetic import geodetic_distance

EQTABLE = OrderedDict([('id', 'integer primary key'),
                       ('sid', 'int'),
                       ('code', 'text'),
                       ('time', 'datetime'),
                       ('latitude', 'float'),
                       ('longitude', 'float'),
                       ('depth', 'float'),
                       ('magnitude', 'float')])
SQTABLE = OrderedDict([('id', 'integer primary key'),
           ('name', 'text'),
           ('start_time', 'datetime'),
           ('end_time', 'datetime'),
           ('center_lat', 'float'),
           ('center_lon', 'float'),
           ('nearest_city', 'text'),
           ('dist_nearest_city', 'float'),
           ('sequence_ended', 'boolean DEFAULT 0'),
           ('n_earthquakes', 'integer'),
           ('projstr', 'text'),
           ('radius', 'float'),
           ('xmin', 'float'),
           ('xmax', 'float'),
           ('ymin', 'float'),
           ('ymax', 'float')])

TABLES={'earthquake':EQTABLE,
          'sequence': SQTABLE}

TIMEFMT='%Y-%m-%dT%H:%M:%S'
TIMEFMT2 = '%Y-%m-%d %H:%M:%S.%f'


class SequenceDatabase(object):
    def __init__(self, dbfile, config, create=False):
        self._db, self._cursor=get_connection_objects(dbfile)
        if create:
            self._createTables()
        self._config=config

    def _createTables(self):
        """
        Build the database tables.
        """
        for table in TABLES.keys():
            sql='CREATE TABLE %s (' % table
            nuggets=[]
            for column, ctype in TABLES[table].items():
                nuggets.append('%s %s' % (column, ctype))
            sql += ','.join(nuggets) + ')'
            self._cursor.execute(sql)

        self._db.commit()
        return

    def mergeSequences(self):
        query = 'SELECT id, xmin, xmax, ymin, ymax FROM sequence WHERE sequence_ended=0'
        df = pd.read_sql_query(query, self._db)
        
        id1_array = []
        id2_array = []

        for i in range(0, len(df)):
            xmin1 = df['xmin'].iloc[i]
            xmax1 = df['xmax'].iloc[i]
            ymin1 = df['ymin'].iloc[i]
            ymax1 = df['ymax'].iloc[i]
            id1 = df['id'].iloc[i]
            
            for j in range(0, len(df)):
                if i==j:
                    continue
                xmin2 = df['xmin'].iloc[j]
                xmax2 = df['xmax'].iloc[j]
                ymin2 = df['ymin'].iloc[j]
                ymax2 = df['ymax'].iloc[j]
                id2 = df['id'].iloc[j]

                bounds1 = (xmin1, xmax1, ymin1, ymax1)
                bounds2 = (xmin2, xmax2, ymin2, ymax2)
                if boxes_intersect(bounds1, bounds2):
                    if id1 not in id2_array and id2 not in id1_array:
                        id1_array.append(id1)
                        id2_array.append(id2)
        sequence_sets = merge_sequences(id1_array, id2_array)
        return sequence_sets


    def getElapsedDataframe(self):
        query='SELECT id, start_time, end_time, n_earthquakes FROM sequence'
        dataframe=pd.read_sql_query(query, self._db, parse_dates=[
            'start_time', 'end_time'])
        dataframe['elapsed']=dataframe['end_time'] - dataframe['start_time']
        return dataframe

    def getStoppedDataframe(self):
        dataframe=self.getElapsedDataframe()

        # get the most recent update time by looking at most EQ loaded
        query2='SELECT max(time) FROM earthquake'
        self._cursor.execute(query2)
        maxtime=datetime.strptime(self._cursor.fetchone()[0], TIMEFMT2)

        # mark sequences that have stopped
        dataframe['gap_days']=maxtime - dataframe['end_time']
        daygap=timedelta(days=self._config['DAYGAP'])
        ended=dataframe['gap_days'] > daygap
        dataframe['ended']=ended
        dataframe = dataframe[dataframe['ended'] == True].copy()

        return dataframe

    def getNonSequences(self):
        dataframe=self.getStoppedDataframe()

        # if we don't have any stopped sequences, bail out
        if not len(dataframe):
            return dataframe

        # delete small/short sequences
        c1=dataframe['n_earthquakes'] < self._config['NUMEQ']
        c2=dataframe['elapsed'] < timedelta(days=self._config['NUMDAYS'])
        deletes=dataframe[c1 | c2]

        return deletes

    def deleteNonSequences(self, deletes):
        # clean out non sequences
        for _, row in deletes.iterrows():
            edelete='DELETE FROM earthquake WHERE sid=%i' % row['id']
            self._cursor.execute(edelete)
            self._db.commit()
            sdelete='DELETE FROM sequence WHERE id=%i' % row['id']
            self._cursor.execute(sdelete)
            self._db.commit()

    def markStoppedSequences(self, keeps):
        # mark dead sequences
        for _, row in keeps.iterrows():
            smark='UPDATE sequence SET sequence_ended=1 WHERE id=%i' % row['id']
            self._cursor.execute(smark)
            self._db.commit()

    def getNumSequences(self):
        query='SELECT count(*) FROM sequence'
        self._cursor.execute(query)
        nseq=self._cursor.fetchone()[0]
        return nseq

    def checkSequence(self, sqstats):
        clat=sqstats['center_lat']
        clon=sqstats['center_lon']
        radius=sqstats['radius']
        seq_query='SELECT id, name, center_lat, center_lon, radius FROM sequence WHERE sequence_ended = 0'
        self._cursor.execute(seq_query)
        seq_rows=self._cursor.fetchall()
        if not len(seq_rows):
            return None
        for row in seq_rows:
            sid=row['id']
            center_lat=row['center_lat']
            center_lon=row['center_lon']
            sradius=row['radius']
            # what is the distance in km from one center to another?
            dist=geodetic_distance(clat, clon, center_lat, center_lon)
            # cmp_dist=max(radius + sradius, self._config['EQDIST2'])
            cmp_dist = radius + sradius
            if dist < cmp_dist:
                return sid

        return None

    def updateSequence(self, dataframe, sqstats, seqid):
        # dataframe is the list of new events...
        # sqstats is the stats for the combined events
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
        query = 'UPDATE sequence SET %s WHERE id=%i' % (colstr, seqid)
        self._cursor.execute(query)
        self._db.commit()
        self.insertEvents(dataframe, seqid)

    def deleteSequence(self, seqid):
        query = 'DELETE FROM sequence WHERE id=%i' % seqid
        self._cursor.execute(query)
        self._db.commit()

    def getSequenceEvents(self, seqid):
        cols = list(EQTABLE.keys())
        colstr = ','.join(cols)
        query = 'SELECT %s FROM earthquake WHERE sid=%i' % (colstr, seqid)
        dataframe = pd.read_sql_query(query, self._db, parse_dates=['time'])
        dataframe = dataframe.drop(['id','sid'],axis=1)
        dataframe = dataframe.rename({'code':'id'}, axis=1)
        return dataframe

    def insertEvents(self, dataframe, seqid):
        for _, row in dataframe.iterrows():
            colstr='(sid, code, time, latitude, longitude, depth, magnitude)'
            valstr='(%i, "%s", "%s", %.4f, %.4f, %.1f, %.1f)'
            tpl=(seqid, row['id'], row['time'],
                   row['latitude'], row['longitude'],
                   row['depth'], row['magnitude'])
            query3='INSERT INTO earthquake %s VALUES %s' % (
                colstr, valstr % tpl)
            self._cursor.execute(query3)
            self._db.commit()

    def insertSequence(self, dataframe, sqstats):
        cols=['name', 'start_time', 'end_time',
                'center_lat', 'center_lon', 'nearest_city',
                'dist_nearest_city',
                'n_earthquakes', 'radius', 'xmin', 'xmax',
                'ymin', 'ymax'
                ]
        colstr=','.join(cols)
        fmt='"%s", "%s", "%s", %.4f, %.4f, "%s", %.2f, %i, %.1f, %.4f, %.4f, %.4f, %.4f'
        tpl=(sqstats['name'],
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
        valstr=fmt % tpl
        query='INSERT INTO sequence (%s) VALUES (%s)' % (colstr, valstr)
        self._cursor.execute(query)
        self._db.commit()
        query2='SELECT last_insert_rowid()'
        seqid=self._cursor.execute(query2).fetchone()[0]

        self.insertEvents(dataframe, seqid)

        return seqid

    def getSequenceNames(self, confirmed=True):
        query = 'SELECT name FROM sequence'
        if confirmed:
            query += ' WHERE sequence_ended = 1'
        self._cursor.execute(query)
        rows = self._cursor.fetchall()
        names = [row[0] for row in rows]
        return names

    def getSequence(self, seqname):
        # 'id', 'integer primary key'),
        #    ('name', 'text'),
        #    ('start_time', 'datetime'),
        #    ('end_time', 'datetime'),
        #    ('center_lat', 'float'),
        #    ('center_lon', 'float'),
        #    ('nearest_city', 'text'),
        #    ('dist_nearest_city', 'float'),
        #    ('sequence_ended', 'boolean DEFAULT 0'),
        #    ('n_earthquakes', 'integer'),
        #    ('projstr', 'text'),
        #    ('radius', 'float'),
        #    ('xmin', 'float'),
        #    ('xmax', 'float'),
        #    ('ymin', 'float'),
        #    ('ymax', 'float')
        cols = list(SQTABLE.keys())
        colstr = ','.join(cols)
        query = 'SELECT %s FROM sequence WHERE name="%s"' % (colstr, seqname)
        df = pd.read_sql_query(query, self._db)
        sequence = df.iloc[0]
        sid = sequence['id']
        sequence = sequence.drop('id').to_dict()
        seqframe = self.getSequenceEvents(sid)
        return (sequence, seqframe)


def merge_sequences(id1, id2):
    pairs = list(zip(id1, id2))
    sets = []
    while len(pairs):
        npairs = len(pairs)
        deletes = [0]
        set1 = set(pairs[0])
        for j in range(1, npairs):
            set2 = set(pairs[j])
            if len(set1.intersection(set2)):
                set1 = set1.union(set2)
                deletes.append(j)
        newpairs = []
        for k in range(npairs):
            if k in deletes:
                continue
            newpairs.append(pairs[k])
        npairs = len(newpairs)
        pairs = newpairs.copy()
        sets.append(set1)
    return sets
                    

    # for i in range(0, len(pairs)):
    #     tset = set()
    #     set1 = set(pairs[i])
    #     for j in range(0, len(pairs)):
    #         if i==j:
    #             continue
    #         set2 = set(pairs[j])
    #         if len(set1.intersection(set2)):
    #             if len(tset):
    #                 tset = tset.union(set2)
    #             else:
    #                 tset = tset.union(set1)
    #                 tset = tset.union(set2)
    #     if len(tset):
    #         sets.append(tset)
    # return sets

def boxes_intersect(bounds1, bounds2):
    xmin1, xmax1, ymin1, ymax1 = bounds1
    xmin2, xmax2, ymin2, ymax2 = bounds2

    coords1 = [(xmin1, ymax1),
              (xmax1, ymax1),
              (xmax1, ymin1),
              (xmin1, ymin1)]
    coords2 = [(xmin2, ymax2),
              (xmax2, ymax2),
              (xmax2, ymin2),
              (xmin2, ymin2)]
    rect1 = Polygon(coords1)
    rect2 = Polygon(coords2)
    if rect1.intersects(rect2):
        return True
  
    return False


def get_connection_objects(dbfile):
    db=sqlite3.connect(dbfile)
    db.row_factory=sqlite3.Row
    cursor=db.cursor()
    return (db, cursor)
