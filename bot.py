import json
from datetime import datetime
import time
from glob import glob
import re
import os

from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterConnectionError, TwitterRequestError
import pandas as pd
from sexmachine.detector import Detector as GenderDetector
import shapefile
from matplotlib.path import Path
import numpy as np
from requests.exceptions import ChunkedEncodingError

# from coord_converter import OSGB36toWGS84

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

import dropbox

app = Flask(__name__)
try:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
    print 'Using database specified in environment variable'
except KeyError:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    print 'Using sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime)
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    handle = db.Column(db.Unicode(100))
    text = db.Column(db.Unicode(500))
    gender = db.Column(db.String(1))
    location = db.Column(db.String(100))

    def __init__(self, timestamp, category, sentiment, handle, text, gender,
                 location):
        self.timestamp = timestamp
        self.category = category
        self.sentiment = sentiment
        self.handle = handle
        self.text = text
        self.gender = gender
        self.location = location

    # def __repr__(self):
    #     return '<User %r>' % self.username

class CountDateTime(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime)
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    n_tweet = db.Column(db.Integer)

    def __init__(self, date_time, category, sentiment):
        self.date_time = date_time
        self.category = category
        self.sentiment = sentiment
        self.n_tweet = 0

class CountLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    n_tweet = db.Column(db.Integer)

    def __init__(self, location, category, sentiment):
        self.location = location
        self.category = category
        self.sentiment = sentiment
        self.n_tweet = 0

class FinalCountLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    n_tweet = db.Column(db.Integer)

    def __init__(self, location, category, sentiment, n_tweet):
        self.location = location
        self.category = category
        self.sentiment = sentiment
        self.n_tweet = n_tweet

class ResultLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    result = db.Column(db.Float)

    def __init__(self, location, category, sentiment, result):
        self.location = location
        self.category = category
        self.sentiment = sentiment
        self.result = result

class CountGender(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    gender = db.Column(db.String(1))
    category = db.Column(db.String(50))
    sentiment = db.Column(db.String(50))
    n_tweet = db.Column(db.Integer)

    def __init__(self, gender, category, sentiment):
        self.gender = gender
        self.category = category
        self.sentiment = sentiment
        self.n_tweet = 0

MODELS = {
    'date_time': CountDateTime,
    'location': CountLocation,
    'gender': CountGender,
}

ALL_MODELS = (CountDateTime, CountLocation, CountGender, ResultLocation,
              FinalCountLocation, Tweet)

with open('hashtags.json') as f_h:
    HASHTAGS = json.load(f_h)

def check_for_hashtags(tweet, hashtag_list):
    for hashtag in hashtag_list:
        if hashtag in tweet['text'].lower():
            return True
    return False

def check_sentiment(tweet, hashtag_dict):
    sentiment = None
    for side, hashtag_list in hashtag_dict.items():
        if check_for_hashtags(tweet, hashtag_list):
            if sentiment is None:
                # Found a sentiment
                sentiment = side
            else:
                # Multiple sentiments; abort
                return None
    return sentiment

def classify_tweet_single(tweet, hashtag_dict):
    tweet_series = [tweet]
    sentiment = None
    while sentiment is None and tweet_series:
        tweet_to_check = tweet_series.pop(0)
        if 'quoted_status' in tweet_to_check:
            tweet_series.append(tweet_to_check['quoted_status'])
        if 'retweeted_status' in tweet_to_check:
            tweet_series.append(tweet_to_check['retweeted_status'])
        sentiment = check_sentiment(tweet_to_check, hashtag_dict)
    result = {
        'sentiment': sentiment,
        'text': tweet_to_check['text'] if sentiment is not None else None,
        }
    return result

def classify_tweet(tweet):
    if 'limit' in tweet:
        return 'limit'
    if 'text' in tweet:
        result = {
            category: classify_tweet_single(tweet, hashtag_dict)
            for category, hashtag_dict in HASHTAGS.items()
        }
        return result
    return 'failed'

def load_creds(name, cred_names, path):
    try:
        creds = {
            key: os.environ[(name + '_' + key).upper()]
            for key in cred_names
        }
    except KeyError:
        with open(path) as f_creds:
            creds = json.load(f_creds)
    return creds

def load_twitter_creds():
    cred_names = ('consumer_key', 'consumer_secret',
                  'access_token', 'access_secret')
    return load_creds('twitter', cred_names, 'twitter-creds.json')

def load_dropbox_creds():
    return load_creds('dropbox', ('token', ), 'dropbox-creds.json')

class Bot(object):

    handle = 'thebrexitbot'

    def __init__(self, test=False):
        print 'Loading Twitter credentials'
        self.twitter_creds = load_twitter_creds()
        print 'Connecting to Twitter'
        self.twitter_api = TwitterAPI(
            self.twitter_creds['consumer_key'],
            self.twitter_creds['consumer_secret'],
            self.twitter_creds['access_token'],
            self.twitter_creds['access_secret'])
        print 'Loading regions'
        self.regions = self.load_regions_us()
        print 'Checking if database has been initialised'
        for model in ALL_MODELS:
            model.__table__.create(db.session.bind, checkfirst=True)
        self.in_db = self.check_db_contents()
        self.final_count_recorded = np.unique(
            [r.location for r in FinalCountLocation.query.all()
             if r.location not in (None, 'other')])
        self.cutoff_times = self.load_cutoff_times()
        self.next_cutoff = self.get_next_cutoff()
        print 'Time to next polls closing: {:.2f} hrs'.format(
            (self.next_cutoff - datetime.now()).total_seconds() / 3600.0)
        print 'Loading names'
        self.gender_detector = GenderDetector()
        self.test = test

    @staticmethod
    def load_regions_uk():
        constituencies = shapefile.Reader('geography/uk/Data/GB/westminster_const_region')
        shapes = constituencies.shapes()
        records = constituencies.records()
        regions = {}
        for shape, record in zip(shapes, records):
            region = record[0][:record[0][:-6].rindex(' ')]
            if region not in regions:
                n_points = len(shape.points)
                if n_points > 200:
                    index = ((np.arange(200) / 199.0) * (n_points - 1)).astype(int)
                else:
                    index = range(n_points)
                points = np.array(shape.points)[index, :]
                points = np.array(
                    [OSGB36toWGS84(easting, northing)
                     for easting, northing in points])
                regions[region] = Path(points)
        return regions

    def load_regions_us(self):
        state_shapes = shapefile.Reader('geography/us/cb_2015_us_state_20m')
        states = {}
        for shape, record in zip(state_shapes.shapes(), state_shapes.records()):
            state = record[4]
            points_list = self.split_shape(np.array(shape.points))
            if state == 'AK':
                points_list = self.simplify_alaska(points_list)
            states[state] = [Path(points) for points in points_list]
        return states

    @staticmethod
    def split_shape(points):
        result = []
        while len(points) > 0:
            stop = np.where(np.all(points[0, :][None, :] == points[1:, :], axis=1))[0][0] + 2
            result.append(points[:stop])
            points = points[stop:]
        return result

    @staticmethod
    def simplify_alaska(points_list):
        points_list = [
            p for p in points_list if np.max(p[:, 0]) > -141.0 and np.max(p[:, 0]) < 0]
        points_list.append(np.array([[-141.0, 50.0],
                                     [-141.0, 75.0],
                                     [-190.0, 75.0],
                                     [-190.0, 50.0],
                                     [-141.0, 50.0]]))
        return points_list

    @staticmethod
    def load_cutoff_times():
        with open('cutoff-times.json') as f_in:
            cutoff_times = json.load(f_in)
        cutoff_times = {key: datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                        for key, value in cutoff_times.items()}
        return cutoff_times

    def get_next_cutoff(self):
        next_cutoff = min([t for t in self.cutoff_times.values() if t > datetime.now()])
        return next_cutoff

    @staticmethod
    def check_db_contents():
        contents = {}
        for key, model in MODELS.items():
            entries = model.query.all()
            for entry in entries:
                if entry.category not in contents:
                    contents[entry.category] = {}
                if entry.sentiment not in contents[entry.category]:
                    contents[entry.category][entry.sentiment] = {}
                if key not in contents[entry.category][entry.sentiment]:
                    contents[entry.category][entry.sentiment][key] = []
                if getattr(entry, key) not in contents[
                        entry.category][entry.sentiment][key]:
                    contents[entry.category][entry.sentiment][key].append(
                        getattr(entry, key))
        return contents

    def get_timestamp(self):
        return self.timestamps[self.timestamps <= datetime.now()][-1]

    def get_gender(self, tweet):
        name = tweet['user']['name']
        if ' ' in name:
            name = name[:name.index(' ')]
        gender = self.gender_detector.get_gender(name, u'great_britain')
        if gender.startswith('mostly_'):
            gender = gender[7:]
        gender = None if gender == 'andy' else gender[0].upper()
        return gender

    def get_region(self, tweet):
        if 'coordinates' in tweet and tweet['coordinates']:
            coordinates = tweet['coordinates']['coordinates']
        elif 'place' in tweet and tweet['place']:
            coordinates = np.mean(
                tweet['place']['bounding_box']['coordinates'], (0, 1))
        else:
            return None
        for region, path_list in self.regions.items():
            for path in path_list:
                if path.contains_point(coordinates):
                    return region
        return 'other'

    def process_tweet(self, tweet):
        if '@thebrexitbot' in tweet['text']:
            self.converse(tweet)
        result = classify_tweet(tweet)
        if isinstance(result, dict):
            region = self.get_region(tweet)
            gender = self.get_gender(tweet)
            for category, cat_result in result.items():
                if cat_result['sentiment'] is not None:
                    self.add_to_db(category, cat_result, tweet, gender, region)
                    if (Tweet.query.count() % 100) == 0:
                        print Tweet.query.count(), 'tweets'
                    if Tweet.query.count() >= 5000:
                        self.dump_tweets()
            # Check if we need to close any polls
            if self.next_cutoff < datetime.now():
                self.close_polls()

    def add_to_db(self, category, cat_result, tweet, gender, region):
        tweet_entry = Tweet(
            datetime.now(), category, cat_result['sentiment'],
            tweet['user']['screen_name'], cat_result['text'],
            gender, region)
        db.session.add(tweet_entry)
        sentiment = cat_result['sentiment']
        now = datetime.now()
        details = {
            'date_time': datetime(now.year, now.month, now.day, now.hour),
            'location': region,
            'gender': gender,
        }
        for key, value in details.items():
            new_value = False
            if category not in self.in_db:
                self.in_db[category] = {}
                new_value = True
            if sentiment not in self.in_db[category]:
                self.in_db[category][sentiment] = {}
                new_value = True
            if key not in self.in_db[category][sentiment]:
                self.in_db[category][sentiment][key] = []
                new_value = True
            if value not in self.in_db[category][sentiment][key]:
                self.in_db[category][sentiment][key].append(value)
                new_value = True
            if new_value:
                entry = MODELS[key](value, category, sentiment)
                db.session.add(entry)
            else:
                kwargs = {
                    key: value,
                    'category': category,
                    'sentiment': sentiment
                }
                entry = MODELS[key].query.filter_by(**kwargs).first()
            entry.n_tweet += 1
        db.session.commit()

    def converse(self, tweet):
        """Process a tweet to the bot and reply if necessary."""
        # if tweet['user']['screen_name'] != 'j_t_allen':
        #     print "Don't want to talk to", tweet['user']['screen_name']
        #     return
        pattern_count = r'[Cc]ount (?P<group_name>.+?) (?P<group_value>.+?)\b'
        match_count = re.search(pattern_count, tweet['text'])
        if match_count:
            group_name = match_count.group('group_name').lower()
            group_value = match_count.group('group_value')
            self.summarise('president', group_name, group_value,
                           at=tweet['user']['screen_name'])
        pattern_result = r'[Rr]esult (?P<location>.+?) (?P<sentiment>.+?) (?P<result>\d+(\.\d+)?)\b'
        match_result = re.search(pattern_result, tweet['text'])
        if match_result and tweet['user']['screen_name'] == 'j_t_allen':
            location = match_result.group('location').upper()
            sentiment = match_result.group('sentiment').lower()
            result = float(match_result.group('result'))
            entry = ResultLocation(location, 'president', sentiment, result)
            db.session.add(entry)
            db.session.commit()
            self.post_tweet('Confirm: {} {} {}'.format(location, sentiment, result), at='j_t_allen')

    def summarise(self, category, group_name, group_value, at=None):
        result = self.get_tweet_frac(category, group_name, group_value)
        if group_name == 'gender':
            screen_value = {'M': 'men', 'F': 'women'}[group_value]
        else:
            screen_value = group_value
        header = 'Twitter sentiment at {} for {}: '.format(
            datetime.now().strftime('%H:%M'), screen_value)
        text = header + ' '.join(
            '{}: {:.1%}'.format(sentiment.title(), frac)
            for sentiment, frac in result.items())
        self.post_tweet(text, at=at)

    def predict(self, state, at=None):
        prediction = self.get_prediction(state)
        header = 'Predicted result at {} for {}: '.format(
            datetime.now().strftime('%H:%M'), state)
        text = header + ' '.join(
            '{}: {:.1%}'.format(sentiment.title(), frac)
            for sentiment, frac in prediction.items())
        self.post_tweet(text, at=at)

    def get_prediction(self, state):
        tweet_frac = self.get_tweet_frac('president', 'location', state)
        if len(tweet_frac) == 0:
            self.post_tweet('No results for '+state, at=at)
            return
        all_results = ResultLocation.query.all()
        all_results = [r for r in all_results if r.location and r.location != 'other']
        states_with_results = np.unique([r.location for r in all_results])
        all_sentiments = np.unique([r.sentiment for r in all_results])
        results_table = pd.DataFrame(index=states_with_results, columns=all_sentiments)
        tweets_table = results_table.copy()
        for row in all_results:
            results_table.loc[row.location, row.sentiment] = row.result
            final_count = FinalCountLocation.query.filter_by(
                category='president', location=row.location, sentiment=row.sentiment).first()
            tweets_table.loc[row.location, row.sentiment] = float(final_count.n_tweet) if final_count else 0.0
        keep = np.all(np.isfinite(results_table.values.astype(float)), axis=1)
        results_table = results_table[keep]
        tweets_table = tweets_table[keep]
        tweets_table.values[tweets_table.values == 0] = 1    # To avoid divide by zero errors
        results_ratio = (results_table[results_table.columns[0]].values /
                         results_table[results_table.columns[1]].values)
        tweets_ratio = (tweets_table[tweets_table.columns[0]].values /
                        tweets_table[tweets_table.columns[1]].values)
        weight = np.sqrt(tweets_table.sum(1).values)
        slope = np.sum(weight * results_ratio / tweets_ratio) / np.sum(weight)
        tweet_frac = [tweet_frac.get(c, 0.0) for c in results_table.columns]
        predicted_ratio = slope * tweet_frac[0] / tweet_frac[1]
        prediction = np.array([predicted_ratio / (1 + predicted_ratio), 1 / (1 + predicted_ratio)])
        prediction = {c: p for c, p in zip(results_table.columns, prediction)}
        return prediction

    def get_tweet_frac(self, category, group_name, group_value):
        n_tweet = self.get_n_tweet(category, group_name, group_value)
        n_all = float(np.sum(n_tweet.values()))
        return {sentiment: n_this / n_all
                for sentiment, n_this in n_tweet.items()}

    def get_n_tweet(self, category, group_name, group_value):
        kwargs = {'category': category, group_name: group_value}
        rows = MODELS[group_name].query.filter_by(**kwargs)
        result = {row.sentiment: row.n_tweet for row in rows}
        return result

    def close_polls(self):
        """Record final counts in closed states."""
        for state, cutoff in self.cutoff_times.items():
            if state not in self.final_count_recorded and cutoff < datetime.now():
                self.final_count_recorded = np.hstack((self.final_count_recorded, state))
                for category in self.in_db:
                    for sentiment in self.in_db[category]:
                        kwargs = {
                            'category': category,
                            'sentiment': sentiment,
                            'location': state
                        }
                        result = MODELS['location'].query.filter_by(**kwargs).first()
                        if result is not None:
                            print 'adding', state, category, sentiment, result.n_tweet
                            entry = FinalCountLocation(
                                state, category, sentiment, result.n_tweet)
                            db.session.add(entry)
        db.session.commit()
        self.next_cutoff = self.get_next_cutoff()



    # def next_timestamp(self):
    #     text = 'Overall sentiment: {:.0%} for Brexit'.format(
    #         self.fraction_for(self.results))
    #     text += '\nIn last 10 mins: {:.0%} for Brexit'.format(
    #         self.fraction_for(self.results.loc[self.current_timestamp]))
    #     self.post_tweet(text)
    #     text = 'By gender:'
    #     for gender in self.genders:
    #         if gender == 'unknown':
    #             continue
    #         text += '\n{}: {:.0%} for Brexit'.format(
    #             gender.title(),
    #             self.fraction_for(
    #                 self.results.loc[pd.IndexSlice[:, gender], :]))
    #     self.post_tweet(text)
    #     text = 'By region:'
    #     for region in self.region_names[:6]:
    #         if region == 'unknown':
    #             continue
    #         text += '\n{}: {:.0%}'.format(
    #             region,
    #             self.fraction_for(
    #                 self.results.loc[pd.IndexSlice[:, :, region], :]))
    #     self.post_tweet(text)
    #     text = ''
    #     for region in self.region_names[6:]:
    #         if region == 'unknown':
    #             continue
    #         text += '\n{}: {:.0%}'.format(
    #             region,
    #             self.fraction_for(
    #                 self.results.loc[pd.IndexSlice[:, :, region], :]))
    #     text = text[1:]
    #     self.post_tweet(text)
    #     self.current_timestamp = self.get_timestamp()
    #     self.results.to_csv('results.csv')
    #     with open(self.get_next_filename(), 'w') as f_out:
    #         json.dump(self.tweets, f_out)
    #     self.tweets = []

    def post_tweet(self, text, at=None):
        if at:
            text = '@' + at + ' ' + text
        print '{} characters'.format(len(text))
        print text
        if not self.test:
            self.twitter_api.request('statuses/update', {'status': text})

    @staticmethod
    def get_next_filename():
        name = 'tweets_{}.json'
        existing = glob(name.format('*'))
        idx = 0
        while name.format(idx) in existing:
            idx += 1
        return name.format(idx)

    @staticmethod
    def get_tweets_path():
        return time.strftime('tweets-%Y%m%d-%H/tweets-%Y%m%d-%H%M%S.json')

    def write_to_dropbox(self, filename, contents):
        """Write a file to Dropbox."""
        print 'Loading Dropbox credentials'
        token = load_dropbox_creds()['token']
        dbx = dropbox.Dropbox(token)
        print 'Uploading to Dropbox'
        if not filename.startswith('/'):
            filename = '/' + filename
        dbx.files_upload(contents, filename)

    def write_to_drive(self, filename, contents):
        """Write a file to Google Drive."""
        print 'Loading Google Drive credentials'
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile('creds.json')
        drive = GoogleDrive(gauth)
        tweets_file = drive.CreateFile(
            {'title': filename,
             'mimeType': 'application/json',
             'shareable': True,
             'userPermission': [
                 {'kind': 'drive#permission',
                  'type': 'anyone',
                  'value': 'anyone',
                  'role': 'reader'}]})
        tweets_file.SetContentString(contents)
        print 'Uploading to drive'
        tweets_file.Upload()

    def dump_tweets(self):
        """Dump all tweets from the database, and clean it."""
        tweets = Tweet.query.all()
        tweets_df = pd.DataFrame({
            key: [getattr(t, key) for t in tweets]
            for key in ('id', 'timestamp', 'category', 'sentiment', 'handle',
                        'text', 'gender', 'location')
            })
        tweets_json = tweets_df.T.to_json()
        if not self.test:
            self.write_to_dropbox(self.get_tweets_path(), tweets_json)
        for tweet in tweets:
            db.session.delete(tweet)
        db.session.commit()

    def download_tweets_drive(self, dir_out='tweets'):
        file_list = self.drive.ListFile({'q': "trashed=false"}).GetList()
        for drive_file in file_list:
            filename = drive_file.metadata['title']
            if (re.match('tweets-\d{8}-\d{6}.json', filename) and
                    filename not in os.listdir(dir_out)):
                drive_file.GetContentFile(os.path.join(dir_out, filename))

    # @staticmethod
    # def fraction_for(results):
    #     return (results.n_out.sum() /
    #             float(results.n_in.sum() + results.n_out.sum()))

    def results_df(self):
        columns = ('category', 'sentiment', 'group_name', 'group_value', 'n_tweet', 'result')
        results = pd.DataFrame(columns=columns)
        for category in self.in_db:
            for sentiment in self.in_db[category]:
                for group_name in self.in_db[category][sentiment]:
                    for group_value in self.in_db[category][sentiment][group_name]:
                        kwargs = {
                            'category': category,
                            'sentiment': sentiment,
                            group_name: group_value
                        }
                        n_tweet = MODELS[group_name].query.filter_by(**kwargs).first().n_tweet
                        row = {
                            'category': category,
                            'sentiment': sentiment,
                            'group_name': group_name,
                            'group_value': group_value,
                            'n_tweet': n_tweet,
                            'result': np.nan
                        }
                        if group_name == 'location':
                            true_result = ResultLocation.query.filter_by(
                                category=category, sentiment=sentiment,
                                location=group_value).first()
                            if true_result is not None:
                                row['result'] = true_result.result
                        results.loc[len(results)] = row
        results['group_value'] = results['group_value'].fillna('unknown')
        return results

    def run(self):
        while True:
            try:
                stream = self.open_stream()
                for tweet in stream:
                    if 'text' in tweet:
                        self.process_tweet(tweet)
                    elif 'disconnect' in tweet:
                        event = tweet['disconnect']
                        if event['code'] in [2,5,6,7]:
                            # something needs to be fixed before re-connecting
                            raise Exception(event['reason'])
                        else:
                            # temporary interruption, re-try request
                            print 'Disconnecting by request of Twitter'
                            break
            except TwitterRequestError as e:
                if e.status_code < 500:
                    # something needs to be fixed before re-connecting
                    print 'Fatal Twitter request error'
                    raise
                else:
                    # temporary interruption, re-try request
                    print 'Non-fatal Twitter request error'
            except TwitterConnectionError:
                # temporary interruption, re-try request
                print 'Non-fatal Twitter connection error'
            except ChunkedEncodingError:
                # not keeping up with the stream, reset it
                print 'Non-fatal chunked encoding error'

    def open_stream(self):
        all_hashtags = []
        for category in HASHTAGS.values():
            for sentiment in category.values():
                all_hashtags.extend(sentiment)
        track = ','.join(all_hashtags + ['@'+self.handle])
        params = {'track': track}
        return self.twitter_api.request('statuses/filter', params)

if __name__ == '__main__':
    Bot().run()
