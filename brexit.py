import json
from datetime import datetime
from glob import glob

from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterConnectionError, TwitterRequestError
import pandas as pd
from sexmachine.detector import Detector as GenderDetector
import shapefile
from matplotlib.path import Path
import numpy as np
from requests.exceptions import ChunkedEncodingError

from coord_converter import OSGB36toWGS84

HASHTAGS_IN = [
    '#'+h for h in ['yes2eu', 'yestoeu', 'betteroffin', 'votein', 'ukineu',
                    'bremain', 'saferin', 'strongerin', 'leadnotleave',
                    'voteremain', 'labourinforbritain', 'catsagainstbrexit',
                    'catsforremain', 'mutts4remain']]
HASHTAGS_OUT = [
    '#'+h for h in ['no2eu', 'notoeu', 'betteroffout', 'voteout', 'eureform',
                    'britainout', 'leaveeu', 'voteleave', 'beleave',
                    'loveeuropeleaveeu', 'leaveeu', 'catsforbrexit']]

def check_for_hashtags(tweet, hashtag_list):
    for hashtag in hashtag_list:
        if hashtag in tweet['text'].lower():
            return True
    return False

def check_sentiment(tweet):
    return (check_for_hashtags(tweet, HASHTAGS_IN),
            check_for_hashtags(tweet, HASHTAGS_OUT))

def classify_tweet(tweet):
    if 'limit' in tweet:
        return 'limit'
    if 'text' in tweet:
        tweet_series = [tweet]
        looks_in, looks_out = False, False
        while not looks_in and not looks_out and tweet_series:
            tweet_to_check = tweet_series.pop(0)
            if 'quoted_status' in tweet_to_check:
                tweet_series.append(tweet_to_check['quoted_status'])
            if 'retweeted_status' in tweet_to_check:
                tweet_series.append(tweet_to_check['retweeted_status'])
            looks_in, looks_out = check_sentiment(tweet_to_check)
        if looks_in and looks_out:
            return 'both'
        elif looks_in:
            return 'in'
        elif looks_out:
            return 'out'
        else:
            return 'neither'
    return 'failed'

class Bot(object):

    def __init__(self):
        print 'Loading credentials'
        with open('creds.json') as f_creds:
            self.creds = json.load(f_creds)
        print 'Connecting to Twitter'
        self.twitter_api = TwitterAPI(
            self.creds['consumer_key'], self.creds['consumer_secret'],
            self.creds['access_token'], self.creds['access_secret'])
        print 'Loading regions'
        self.regions = self.load_regions()
        print 'Loading names'
        self.gender_detector = GenderDetector()
        print 'Preparing for tweets'
        self.timestamps = pd.date_range(datetime.now(), periods=6*48, freq='600S')
        self.genders = ('male', 'female', 'unknown')
        self.region_names = self.regions.keys() + ['non-UK', 'unknown']
        index = pd.MultiIndex.from_product(
            [self.timestamps, self.genders, self.region_names],
            names=('timestamp', 'gender', 'region'))
        self.results = pd.DataFrame(
            {'n_in': np.zeros(len(index)),
             'n_out': np.zeros(len(index)),
             'n_both': np.zeros(len(index)),
             'n_neither': np.zeros(len(index))},
            index=index).sort_index()
        self.tweets = []
        self.current_timestamp = self.timestamps[0]

    def load_regions(self):
        euro_shapes = shapefile.Reader('Data/GB/european_region_region')
        shapes = euro_shapes.shapes()
        records = euro_shapes.records()
        regions = {}
        for shape, record in zip(shapes, records):
            region = record[0][:-12]
            if region not in regions:
                points = np.array(shape.points)[::len(shape.points)/1000, :]
                points = np.array(
                    [OSGB36toWGS84(easting, northing)
                     for easting, northing in points])
                regions[region] = Path(points)
        return regions

    def get_timestamp(self):
        return self.timestamps[self.timestamps <= datetime.now()][-1]

    def get_gender(self, tweet):
        name = tweet['user']['name']
        if ' ' in name:
            name = name[:name.index(' ')]
        gender = self.gender_detector.get_gender(name, u'great_britain')
        if gender.startswith('mostly_'):
            gender = gender[7:]
        elif gender == 'andy':
            gender = 'unknown'
        return gender

    def get_region(self, tweet):
        if 'coordinates' in tweet and tweet['coordinates']:
            coordinates = tweet['coordinates']['coordinates']
        elif 'place' in tweet and tweet['place']:
            coordinates = np.mean(
                tweet['place']['bounding_box']['coordinates'], (0, 1))
        else:
            return 'unknown'
        for region, path in self.regions.items():
            if path.contains_point(coordinates):
                return region
        return 'non-UK'

    def process_tweet(self, tweet):
        self.tweets.append(tweet)
        sentiment = classify_tweet(tweet)
        if sentiment not in ('in', 'out', 'both', 'neither'):
            return
        gender = self.get_gender(tweet)
        region = self.get_region(tweet)
        timestamp = self.get_timestamp()
        if timestamp != self.current_timestamp:
            self.next_timestamp()
        self.results.loc[(timestamp, gender, region), 'n_'+sentiment] += 1

    def next_timestamp(self):
        text = 'Overall sentiment: {:.0%} for Brexit'.format(
            self.fraction_for(self.results))
        text += '\nIn last 10 mins: {:.0%} for Brexit'.format(
            self.fraction_for(self.results.loc[self.current_timestamp]))
        self.post_tweet(text)
        text = 'By gender:'
        for gender in self.genders:
            if gender == 'unknown':
                continue
            text += '\n{}: {:.0%} for Brexit'.format(
                gender.title(),
                self.fraction_for(
                    self.results.loc[pd.IndexSlice[:, gender], :]))
        self.post_tweet(text)
        text = 'By region:'
        for region in self.region_names[:6]:
            if region == 'unknown':
                continue
            text += '\n{}: {:.0%}'.format(
                region,
                self.fraction_for(
                    self.results.loc[pd.IndexSlice[:, :, region], :]))
        self.post_tweet(text)
        text = ''
        for region in self.region_names[6:]:
            if region == 'unknown':
                continue
            text += '\n{}: {:.0%}'.format(
                region,
                self.fraction_for(
                    self.results.loc[pd.IndexSlice[:, :, region], :]))
        text = text[1:]
        self.post_tweet(text)
        self.current_timestamp = self.get_timestamp()
        self.results.to_csv('results.csv')
        with open(self.get_next_filename(), 'w') as f_out:
            json.dump(self.tweets, f_out)
        self.tweets = []

    def post_tweet(self, text):
        print '{} characters'.format(len(text))
        print text
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
    def fraction_for(results):
        return (results.n_out.sum() /
                float(results.n_in.sum() + results.n_out.sum()))

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
        params = {'track': ','.join(HASHTAGS_IN + HASHTAGS_OUT)}
        return self.twitter_api.request('statuses/filter', params)


if __name__ == '__main__':
    stream = twitter_api.request('statuses/filter', {'track': ','.join(hashtags_in + hashtags_out)})

    results = {'limit': 0, 'both': 0, 'in': 0, 'out': 0, 'neither': 0, 'failed': 0, 'geo': 0, 'count': 0, 'coordinates': 0, 'place': 0}

    for tweet in stream:
        r = classify_tweet(tweet)
        results[r] += 1
        results['count'] += 1
        if r == 'failed':
            print tweet
        if 'geo' in tweet and tweet['geo']:
            geo = 1
        else:
            geo = 0
        results['geo'] += geo
        if 'coordinates' in tweet and tweet['coordinates']:
            coordinates = 1
        else:
            coordinates = 0
        results['coordinates'] += coordinates
        if 'place' in tweet and tweet['place']:
            place = 1
        else:
            place = 0
        results['place'] += place
        if results['count'] % 100 == 0:
            print results
        if r == 'neither' or r == 'both' or r == 'failed' or coordinates or geo or place:
            tweet_list.append(tweet)
