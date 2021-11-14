import configparser

import pandas as pd
import tweepy


class TweetManager(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        auth = tweepy.OAuthHandler(config['TWITTER_AUTH']['API_key'], config['TWITTER_AUTH']['API_secret_key'])
        auth.set_access_token(config['TWITTER_AUTH']['Access_token'], config['TWITTER_AUTH']['Access_token_secret'])
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)

    def get_tweets(self, query, result_type, count, lang='ar'):
        tweets = tweepy.Cursor(self.twitter_api.search, q=query, count=count, lang=lang, result_type=result_type)

        '''
        # code pour obtenir le texte complet des tweets
        data2 = []
        for tweet in tweets.items(count):
            id = tweet.id
            try:
                full = self.twitter_api.get_status(id, tweet_mode="extended")
            except:
                print('no status')
            try:
                fullText = full.retweeted_status.full_text
                #print(fullText)
            except AttributeError:  # Not a Retweet
                fullText= full.full_text
                #print(full.full_text)
            #print(tweet.text)
            #print(tweet.user)
            #print(tweet.coordinates)
            data2.append([tweet.created_at, fullText])

        data = []
        for [tweet.created_at, tweet.text] in tweets.items(count):
            data.append([tweet.created_at, tweet.text])
        '''
        data = [[tweet.created_at, tweet.text] for tweet in tweets.items(count)]

        return pd.DataFrame(data, columns=['created_at', 'tweet'])


def main():
    """
    To test the classifier
    """
    #df = TweetManager().get_tweets_dummy('corona', count=100, result_type='popular')
    df = TweetManager().get_tweets('كورونا', count=100, result_type='mixed')
    print(df)

if __name__ == '__main__':
    main()
