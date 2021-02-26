from pathlib import Path
from typing import List, Union, Tuple
import pickle
import re
import pandas as pd
import tweepy
import emoji
import neologdn

import myutil


prj_path = myutil.PROJECT_PATH


def log(msg: Union[str, Path], exception=False) -> str:
    suffix = '[{}] '.format(Path(__name__))
    if isinstance(msg, Path):
        msg = str(msg)

    if exception:
        Exception(suffix + msg)
    else:
        print(suffix + msg)


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
        hololive_members = pd.read_csv(Path(prj_path, 'data/hololive_members.csv'))
        member_info = hololive_members[hololive_members['twitter'] == userId]

        self.userId = member_info.twitter.values[0]  # ユーザID
        self.name = member_info.name.values[0]      # フルネーム
        self.generation = set(member_info.generation.values[0].split('/'))  # 世代（n期/ゲーマーズ）

        self.n_tweet = n_tweet
        self.load = load
        self.verbose = verbose
        self.preview = preview

        # ツイートデータの保存パス
        self.tweets_path = Path(prj_path, 'data/tweets_{0}_{1:d}.pkl'.format(self.userId, self.n_tweet))
        # ツイートデータのデータフレームの保存パス
        self.tweets_df_path = Path(prj_path, 'data/tweets_{0}_{1:d}.tsv'.format(self.userId, self.n_tweet))
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
                log('以下パスよりツイートデータを読み込みました')
                log('{}'.format(self.tweets_path))
        else:
            # APIを利用したツイートデータ取得
            tweets = [
                tweet for tweet in tweepy.Cursor(api.user_timeline,
                                                 screen_name=self.userId, include_rts=True).items(self.n_tweet)]
            if save:
                with open(self.tweets_path, mode='wb') as f:
                    pickle.dump(tweets, f)
                if self.verbose:
                    log('以下パスにツイートデータを保存しました')
                    log('{}'.format(self.tweets_path))

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
        if self.tweets_df_path.exists():
            # 既存データフレーム読み込み
            tweets_df = pd.read_csv(self.tweets_df_path, sep='\t', index_col=0, parse_dates=[2])
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
                tweets_df.to_csv(self.tweets_df_path, sep='\t', index=True)

        if self.preview:
            print(tweets_df.head(5))

        return tweets_df

    def format_tweets_df(self, save=True) -> None:
        """ツイートデータのデータフレームの整形（text列の前処理）

        Args:
            save (bool, optional): データフレームのcsv形式での上書き保存. Defaults to True.
        """
        tweets_df = self.df.copy()

        # RTを削除
        tweets_df_without_rt = tweets_df[tweets_df['text'].str[:2] != 'RT'].copy()
        # ツイートテキスト（text列）整形（詳細はformat_tweet_text()参照）
        tweets_df_without_rt['text'] = tweets_df_without_rt['text'].map(format_tweet_text).copy()
        # 内容がないツイート削除
        tweets_df_without_empty = tweets_df_without_rt[~tweets_df_without_rt['text'].isin([' ', ''])].copy()
        # データフレーム更新
        self.df = tweets_df_without_empty

        if save:
            tweets_df_without_empty.to_csv(self.tweets_df_path, sep='\t', index=True)

        if self.preview:
            print(tweets_df.head(5))


def format_tweet_text(tweet_text: str, return_original=False) -> Union[str, Tuple[str, str]]:
    """ツイートテキスト整形

    Args:
        tweet_text (str): ツイートテキスト
        return_original (bool, optional): 整形前のツイートテキストも返す. Defaults to False.

    Returns:
        Union[str, Tuple[str, str]]: 整形後ツイートテキスト（return_original=Trueで整形前のツイートテキストも返す）
    """
    # URL除去
    formatted_tweet_text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', tweet_text)
    # 絵文字除去
    formatted_tweet_text = ''.join(['' if c in emoji.UNICODE_EMOJI['en'] else c for c in formatted_tweet_text])
    # 正規化
    formatted_tweet_text = neologdn.normalize(formatted_tweet_text)
    # リプライ先（＠ユーザ名）の除去
    formatted_tweet_text = re.sub(r'@[ ]?[a-zA-z0-9]+', '', formatted_tweet_text)
    # ハッシュタグ除去
    formatted_tweet_text = re.sub(r'[#][A-Za-z一-鿆0-9ぁ-ヶｦ-ﾟー]+', '', formatted_tweet_text)
    # 改行除去
    formatted_tweet_text = re.sub(r'[\n]', ' ', formatted_tweet_text)
    # 半角記号除去
    formatted_tweet_text = re.sub(r'[!\"\#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\\]', ' ', formatted_tweet_text)
    # 全角記号除去
    formatted_tweet_text = re.sub(r'[・〜＋｜’；：＞＜」「（）％＄＃＠＆＾＊！？【】『』＼／…○△□●▲■▼▶◀▽★☆※‥]',
                                  '', formatted_tweet_text)
    # 数字除去
    formatted_tweet_text = re.sub(r'[0-9]', '', formatted_tweet_text)
    # 除草
    formatted_tweet_text = re.sub(r'[w]{2,}', ' ', formatted_tweet_text)
    # 特殊記号除去
    formatted_tweet_text = re.sub(r'[ก˶᷇ ᷆˵ᯅʅʃ▓ु▷§٩งᐠ•๑ว≡￣⃕ᐟ𓏸◝ཫᆺ́ㅅεψζ∠◆༥³♡̩˘◥￢ᔦ✿﹀◦⌓┃₍₍⁾⁾  ृ๛ ͟͟͞ᔨ⸜ㅂ◤꒳。◟ﾟ｀ʖ└ᴗ´⋆˚ଘ⸝╹│Ꙭ̮𓂂▹▸°↝̀͜੭ㄘζو̑̑߹〃³¦༥˙ꇤ꜄꜆ ⃔зੈ॑₊ʘ ωᗜ⊂‧ᐛ⌒♆❛ᵕ✩♪◡▀̿ヾ𖥦ᵒ̴̶̷✧ˆˊˋ]',
                                  ' ', formatted_tweet_text)
    # 連続半角スペースをまとめる
    formatted_tweet_text = re.sub(r'[ ]+', ' ', formatted_tweet_text)

    if return_original:
        return formatted_tweet_text, tweet_text
    else:
        return formatted_tweet_text
