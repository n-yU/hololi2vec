from pathlib import Path
from typing import List, Union, Tuple, Dict
import pickle
import re
from datetime import datetime as dt
from tqdm import tqdm

import pandas as pd
import numpy as np
import tweepy
import emoji
import neologdn
import MeCab

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


class Hololive:
    """ホロライブメンバーのクラスHolomemの持つデータを集約する
    """

    def __init__(self, userIds: np.ndarray, n_tweet: int, fname: str) -> None:
        """インスタンス化

        Args:
            userIds (np.ndarray): Twitter ID（ユーザID）
            n_tweet (int): 取得ツイート数
            fname (str): ファイル名接頭辞
        """
        self.userIds = userIds
        self.n_tweet = n_tweet
        self.fname = fname

        # userIdsで指定したホロメンのHolomem.dfを集約した辞書取得（キーはユーザID）
        self.tweets_dfs = self.get_tweets_dfs()
        # tweets_dfsのデータフレームを連結＆時系列順にソートしたデータフレームを取得
        self.all_user_tweets_df = self.get_all_user_tweets_df()

    def get_tweets_dfs(self) -> Dict[str, Holomem]:
        """userIdsで指定したホロメンのHolomem.dfを集約した辞書を取得する

        Returns:
            Dict[str, Holomem]: データフレーム集約辞書
        """
        tweets_dfs = dict()
        for userId in self.userIds:
            holomem = Holomem(userId=userId, n_tweet=self.n_tweet, load=True, verbose=False, preview=False)
            # 時系列ソート -> 欠損行削除 -> インデックスリセット
            tweets_dfs[userId] = holomem.df.iloc[::-1].dropna(how='any', axis=0).reset_index(drop=True)

        return tweets_dfs

    def get_all_user_tweets_df(self) -> pd.core.frame.DataFrame:
        """tweets_dfsのデータフレームを連結＆時系列順にソートしたデータフレームを取得する

        Returns:
            pd.core.frame.DataFrame: 連結データフレーム
        """
        dfs = self.tweets_dfs.values()
        # 時系列ソート -> インデックスリセット
        all_user_tweets_df = pd.concat(dfs, axis=0).sort_values(
            ['timestamp', 'userId'], ascending=[True, True]).reset_index(drop=True)
        return all_user_tweets_df

    def __wakatigaki(self) -> None:
        """[private] ツイートテキストの分かち書きを行う
        """
        # MeCab辞書 -> NEologd
        neologd = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        one_word_tweet_idx = []  # テキストが2単語未満のツイートのインデックス
        all_user_tweets_df = self.all_user_tweets_df.copy()

        # 分かち書き結果ファイルが既に存在 -> 削除して作成し直す
        if self.wakatigaki_path.exists():
            self.wakatigaki_path.unlink()

        with open(self.wakatigaki_path, mode='a', encoding='utf-8') as f:
            for tweet in all_user_tweets_df.itertuples():
                idx, text = tweet.Index, tweet.text
                neologd_text = neologd.parse(text)  # Chasen形式で分かち書き
                # 単語をリスト形式で追加（品詞='記号-一般'・EOF除く）
                wakati_words = [word.split('\t')[0] for word in neologd_text.split('\n')
                                if len(word.split('\t')) == 6 and word.split('\t')[3] != '記号-一般']

                # ツイートを構成する単語数が2未満 -> one_word_tweet_idxに追加
                if len(wakati_words) < 2:
                    one_word_tweet_idx.append(idx)
                    continue

                # 単語ごとにスペース区切り，ツイートごとに改行区切りでファイル書き込み
                wakati_text = ' '.join(wakati_words)
                f.write(wakati_text)
                f.write('\n')

        # one_word_tweet_idxのインデックスに対応する行（ツイート）をall_user_tweets_dfから削除
        all_user_tweets_df.drop(index=one_word_tweet_idx, inplace=True)
        self.all_user_tweets_df = all_user_tweets_df

    def split_data(self, date: dt, save=True, verbose=False) -> Tuple[
            pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        """データフレームを訓練・テストセットに分割する

        Args:
            date (dt): 分割日時
            save (bool, optional): 分割データフレームのtsv形式での保存. Defaults to True.
            verbose (bool, optional): 保存パスなどの情報出力. Defaults to False.

        Returns:
            Tuple[ pd.core.frame.DataFrame, pd.core.frame.DataFrame]: 訓練・テストセット（データフレーム）
        """
        self.change_date(date=date, verbose=False)  # 分割日設定（保存パス指定）
        self.__wakatigaki()                         # ツイートテキストの分かち書き
        all_user_tweets_df = self.all_user_tweets_df.copy()

        # all_user_tweets_dfの分割（分割日時上のツイートはテストセットに入る）
        train_df = all_user_tweets_df[all_user_tweets_df['timestamp'] < date]
        test_df = all_user_tweets_df[all_user_tweets_df['timestamp'] >= date]

        if save:
            train_df.to_csv(self.train_df_path, sep='\t', index=True)
            test_df.to_csv(self.test_df_path, sep='\t', index=True)

        self.train_df = train_df
        self.test_df = test_df

        # 訓練セットに対応するように分かち書き結果ファイルの末尾（テストセット部）をカット
        with open(self.wakatigaki_path, mode='r', encoding='utf-8') as f:
            wakati_text = f.readlines()
        wakati_text_train = ''.join(wakati_text[:train_df.shape[0]])
        with open(self.wakatigaki_path, mode='w', encoding='utf-8') as f:
            f.write(wakati_text_train)

        if verbose:
            log('以下パスに分かち書き結果とツイートデータを保存しました')
            log('分かち書き結果（訓練セット）: {}'.format(self.wakatigaki_path))
            log('訓練セット: {}'.format(self.train_df_path))
            log('テストセット: {}'.format(self.test_df_path))

        return train_df, test_df

    def __get_session_data(self, df: pd.core.frame.DataFrame) -> List[Tuple[int, str, str, dt]]:
        """ツイートデータフレームからセッションデータ形式のリスト（session_data）を取得する

        Args:
            df (pd.core.frame.DataFrame): ツイートデータフレーム

        Returns:
            List[Tuple[int, str, str, dt]]: セッションデータ形式のリスト
        """
        neologd = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        session_data = []

        for tweet in tqdm(df.itertuples(), total=df.shape[0]):
            # 各ツイートを単語ごとに行を分けてログを作成 -> session_dataに追加
            neologd_text = neologd.parse(tweet.text)
            words = [word.split('\t')[0] for word in neologd_text.split('\n')
                     if len(word.split('\t')) == 6 and word.split('\t')[3] != '記号-一般']

            tId, uId, timestamp = tweet.Index, tweet.userId, tweet.timestamp
            tweet_logs = [[tId, uId, word, timestamp] for word in words]
            session_data.extend(tweet_logs)

        return session_data

    def create_session_df(self, train_test=[True, True], save=True, verbose=False
                          ) -> Union[pd.core.frame.DataFrame,
                                     Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]]:
        """セッションデータを作成する

        Args:
            train_test (list, optional): Trueに設定したデータセットからそれぞれセッションデータを作成する. \
                Defaults to [True, True].
            save (bool, optional): セッションデータのcsv形式での保存. Defaults to True.
            verbose (bool, optional): 保存パスなどの情報出力. Defaults to False.

        Returns:
            Union[pd.core.frame.DataFrame, Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]]: \
                train_testの指定に対応するセッションデータ
        """
        train_df, test_df = self.train_df.copy(), self.test_df.copy()
        session_df_columns = ['tweetId', 'userId', 'word', 'timestamp']

        # セッションデータ形式のリストをデータフレーム形式に変換
        if train_test[0]:
            train_session_data = self.__get_session_data(df=train_df)
            train_session_df = pd.DataFrame(train_session_data, columns=session_df_columns)
            self.train_session_df = train_session_df
        if train_test[1]:
            test_session_data = self.__get_session_data(df=test_df)
            test_session_df = pd.DataFrame(test_session_data, columns=session_df_columns)
            self.test_session_df = test_session_df

        if save:
            if train_test[0]:
                train_session_df.to_csv(self.train_session_df_path, index=True)
                if verbose:
                    log('以下パスに訓練セットのセッションデータを保存しました')
                    log(self.train_session_df_path)

            if train_test[1]:
                test_session_df.to_csv(self.test_session_df_path, index=True)
                if verbose:
                    log('以下パスにテストセットのセッションデータを保存しました')
                    log(self.test_session_df_path)

        if train_test[0] and train_test[1]:
            return train_session_df, test_session_df
        elif train_test[0]:
            return train_session_df
        elif train_test[1]:
            return test_session_df
        else:
            log('train_testは必ずどちらかはTrueにしてください', exception=True)

    def __calc_session_len(self, df: pd.core.frame.DataFrame) -> Tuple[float, int, int]:
        """[private] セッションデータのセッション長の平均・最大・最小を計算する

        Args:
            df (pd.core.frame.DataFrame): [description]

        Returns:
            Tuple[float, int, int]: [description]
        """
        # tweetIdでグループ化 -> 各グループのデータフレームのサイズをリストに追加
        sessions_len = np.array([tId_df.shape[0] for _, tId_df in df.groupby('tweetId')])
        return np.mean(sessions_len), np.max(sessions_len), np.min(sessions_len)

    def show_session_len(self) -> None:
        """セッションデータのセッション長の平均・最大・最小の計算結果を表示する
        """
        if self.train_session_df is not None:
            train_session_df = self.train_session_df.copy()
            avg_len, max_len, min_len = self.__calc_session_len(train_session_df)
            log('訓練セットのセッション長情報')
            print('Average:{0} / Max:{1} / Min:{2}'.format(avg_len, max_len, min_len))
        if self.test_session_df is not None:
            test_session_df = self.test_session_df.copy()
            avg_len, max_len, min_len = self.__calc_session_len(test_session_df)
            log('テストセットのセッション長情報')
            print('Average:{0} / Max:{1} / Min:{2}'.format(avg_len, max_len, min_len))

    def change_date(self, date: dt, verbose=False) -> None:
        """分割日時と保存パスを変更する

        Args:
            date (dt): 分割日時
            verbose (bool, optional): 保存パスなどの情報出力. Defaults to False.
        """
        # 変更前の分割日時を保持
        if 'date' in vars(self).keys():
            prv_date = self.date
        else:
            prv_date = None
        self.date = date
        date_str = date.strftime('%Y%m%d')  # datetimeの出力形式指定

        # ファイル名: ${fname}_${n_tweet}_${date}_{train/test}_{session}
        self.wakatigaki_path = Path(prj_path, 'data/wakatigaki/{0}_{1}_{2}.txt'.format(
            self.fname, self.n_tweet, date_str))    # 分かち書き結果 (txt)
        self.train_df_path = Path(prj_path, 'data/dataset/{0}_{1}_{2}_train.tsv'.format(
            self.fname, self.n_tweet, date_str))    # ツイートデータ: 訓練セット (tsv)
        self.test_df_path = Path(prj_path, 'data/dataset/{0}_{1}_{2}_test.tsv'.format(
            self.fname, self.n_tweet, date_str))    # ツイートデータ: テストセット (tsv)
        self.train_session_df_path = Path(prj_path, 'data/session/{}'.format(
            self.train_df_path.stem + '_session.csv'))  # セッションデータ: 訓練セット (csv)
        self.test_session_df_path = Path(prj_path, 'data/session/{}'.format(
            self.test_df_path.stem + '_session.csv'))   # セッションデータ: テストセット (csv)

        if verbose:
            if prv_date is None:
                log('訓練・テストセットの分割日を"{}"に設定しました'.format(date_str))
            else:
                log('訓練・テストセットの分割日を"{0}"から"{1}"に変更しました'.format(
                    prv_date.strftime('%Y%m%d'), date_str))
