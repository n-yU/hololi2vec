import logging
from pathlib import Path
from typing import List
import pandas as pd
import pickle
import tweepy

import myutil

# logging config
# formatter = '[hololi2vec:dataset] %(message)s'
formatter = '[hololi2vec:{}] %(message)s'.format(Path(__file__).stem)
logging.basicConfig(level=logging.INFO, format=formatter)


def verify_tweepy(wait_on_rate_limit=True, wait_on_rate_limit_notify=True) -> tweepy.API:
    """Tweepyによるユーザ認証

    Args:
        wait_on_rate_limit (bool, optional): レート制限の補充をオート待機. Defaults to True.
        wait_on_rate_limit_notify (bool, optional): レート制限の補充待機時の通知. Defaults to True.

    Returns:
        tweepy.API: Twitter API Wrapper
    """
    # APIキー読み込み
    api_key, api_secret, access_token, access_secret = myutil.load_twitter_api_keys()

    # ユーザ認証(OAuth)
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=wait_on_rate_limit,
                     wait_on_rate_limit_notify=wait_on_rate_limit_notify)

    return api


class Holomem:
    """ホロライブメンバー
    """

    def __init__(self, userId: str, n_tweet: int, load=True, verbose=False, preview=False) -> None:
        """インスタンス化

        Args:
            userId (str): Twitter ID（ユーザID）
            n_tweet (int): 取得ツイート数
            load (bool, optional): 既存データを読み込む（予期せぬAPI呼び出し回避のため）. Defaults to True.
            verbose (bool, optional): 保存パスなどの情報出力. Defaults to False.
            preview (bool, optional): 取得・読み込んだデータの先頭部表示（5レコード）. Defaults to False.
        """
        # 指定したユーザIDのメンバー情報取得
        hololive_members = pd.read_csv(Path(myutil.PROJECT_PATH, 'data/hololive_members.csv'))
        member_info = hololive_members[hololive_members['twitter'] == userId]

        self.userId = member_info.twitter.values[0]  # ユーザID
        self.name = member_info.name.values[0]      # フルネーム
        self.generation = set(member_info.generation.values[0].split('/'))  # 世代（n期/ゲーマーズ）

        self.n_tweet = n_tweet
        self.load = load
        self.verbose = verbose
        self.preview = preview

        # ツイートデータの保存パス
        self.tweets_path = Path(myutil.PROJECT_PATH,
                                'data/tweets_{0}_{1:d}.pkl'.format(self.userId, self.n_tweet))
        self.tweets = self.get_tweets(save=True)    # ツイートデータ取得
        self.df = self.create_tweets_df(save=True)  # ツイートデータのデータフレーム作成

    def get_tweets(self, save=True) -> List[tweepy.models.Status]:
        """ツイートデータ取得

        Args:
            save (bool, optional): ツイートデータのpkl形式での保存. Defaults to True.

        Returns:
            List[tweepy.models.Status]: ツイートデータ（ツイートに関するすべてのデータ保持）
        """
        api = verify_tweepy()   # APIユーザ認証

        if self.load or self.tweets_path.exists():
            # 既存ツイートデータの読み込み
            with open(self.tweets_path, mode='rb') as f:
                tweets = pickle.load(f)
            if self.verbose:
                logging.info('以下パスよりツイートデータを読み込みました')
                logging.info('{}'.format(self.tweets_path))
        else:
            # APIを利用したツイートデータ取得
            tweets = [
                tweet for tweet in tweepy.Cursor(api.user_timeline,
                                                 screen_name=self.userId, include_rts=False).items(self.n_tweet)]
            if save:
                with open(self.tweets_path, mode='wb') as f:
                    pickle.dump(tweets, f)
                if self.verbose:
                    logging.info('以下パスにツイートデータを保存しました')
                    logging.info('{}'.format(self.tweets_path))

        if self.preview:
            # 取得できたツイートデータの先頭5つ表示（最新5ツイート）
            for tweet in tweets[:5]:
                print('-------------------------')
                print(tweet.created_at)
                print(tweet.text)

        return tweets

    def create_tweets_df(self, save=True) -> pd.core.frame.DataFrame:
        """ツイートデータのデータフレーム作成

        Args:
            save (bool, optional): データフレームのcsv形式での保存. Defaults to True.

        Returns:
            pd.core.frame.DataFrame: ツイートデータのデータフレーム（ユーザID，投稿日時，テキスト）
        """
        # ツイートデータのデータフレームの保存パス
        tweets_df_path = Path(myutil.PROJECT_PATH,
                              'data/tweets_{0}_{1:d}.tsv'.format(self.userId, self.n_tweet))

        if tweets_df_path.exists():
            # 既存データフレーム読み込み
            tweets_df = pd.read_csv(tweets_df_path, sep='\t', index_col=0)
        else:
            # ツイートデータからデータフレーム作成
            # （ツイートデータのツイート数は上限に達することでn_tweetより少なくなることがあるため，
            # ツイートデータの長さを取得する）
            tweets_data = [[] for _ in range((len(self.tweets)))]
            for idx, tweet in enumerate(self.tweets):
                data = [tweet.user.screen_name, tweet.created_at, tweet.text]
                tweets_data[idx] = data

            tweets_df = pd.DataFrame(tweets_data, columns=['userId', 'timestamp', 'text'])

            if save:
                tweets_df.to_csv(tweets_df_path, sep='\t', index=True)

        if self.preview:
            print(tweets_df.head(5))

        return tweets_df
