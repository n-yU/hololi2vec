from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm.notebook import tqdm as tqdm_nb
from copy import copy, deepcopy
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd
from gensim.models import word2vec

import myutil
from dataset import Holomem, Hololive
from model import ProposalModel, calc_similarity


prj_path = myutil.PROJECT_PATH
DUMMY_DATE = dt(2001, 1, 1)


def log(msg: Union[str, Path], exception=False) -> str:
    suffix = '[{}] '.format(Path(__name__))
    if isinstance(msg, Path):
        msg = str(msg)

    if exception:
        raise Exception(suffix + msg)
    else:
        print(suffix + msg)


class ProposalModelAnalyzer():
    global DUMMY_DATE

    def __init__(self, proposal_model: ProposalModel):
        try:
            self.history = proposal_model.reps_history
        except AttributeError:
            log('テストセットによるモデル学習を先に行ってください', exception=True)

        self.userIds = proposal_model.userIds
        self.words_name = proposal_model.w2v_model.words
        self.words_rep = proposal_model.w2v_model.model.wv

        self.df = proposal_model.test
        self.isLast = proposal_model.get_isLast(isTrain=False)
        self.start_dt = proposal_model.test['timestamp'].min()
        self.end_dt = proposal_model.test['timestamp'].max()

    def merge_period(self, periods: List[np.ndarray]) -> np.ndarray:
        period = periods[0].copy()

        for prd in periods[1:]:
            period = np.intersect1d(period, prd)
        return period

    def get_period_by_date(self, start_dt: dt, end_dt: dt) -> np.ndarray:
        period = self.df[(start_dt <= self.df['timestamp']) & (self.df['timestamp'] <= end_dt)].index.to_numpy()
        return period

    def get_idx_by_date(self, date: dt, days=1) -> int:
        period = self.get_period_by_date(start_dt=date, end_dt=date + timedelta(days=days))
        try:
            idx = period[0] - 1
        except IndexError:
            log('指定日より{}日後のツイートデータを検索しましたが見つかりませんでした'.format(days), exception=True)
            log('引数daysの値を増やして再実行してください', exception=True)
        return idx

    def get_timestamp_by_period(self, period: np.ndarray) -> np.ndarray:
        date = list(map(pd.to_datetime, self.df.loc[period]['timestamp']))
        return date

    def get_period_by_tweet(self, tweetId: int) -> np.ndarray:
        period = self.get_tweet_df(tweetId=tweetId).index.to_numpy()
        return period

    def get_period_by_users(self, userIds: List[str], start_dt: dt, end_dt: dt) -> np.ndarray:
        date_period = self.get_period_by_date(start_dt=start_dt, end_dt=end_dt)
        userIds_set = set(userIds)
        period = []

        for idx in date_period:
            uId = self.df.loc[idx]['userId']
            if (uId in userIds_set) and self.isLast[idx]:
                period.append(idx)

        return period

    def get_tweetId_by_date(self, userId: str, date: dt, days=1) -> int:
        period = self.get_period_by_date(start_dt=date - timedelta(days=days), end_dt=date)
        df = self.df.loc[period]
        userIds = df[df['userId'] == userId]['tweetId'].to_numpy()

        try:
            userId = userIds[-1]
        except IndexError:
            log('指定日より前の{}日間のツイートデータを検索しましたが見つかりませんでした\n \
                引数daysの値を増やして再実行してください'.format(days), exception=True)
        return userId

    def get_rep(self, idx: int, userId: str, rep_type: str) -> Tuple[int, np.ndarray]:
        rep = self.history[idx][userId][rep_type]
        return rep

    def get_rep_list(self, period: np.ndarray, userId: str, rep_type: str) -> Dict[int, np.ndarray]:
        rep_list = dict()

        for idx in period:
            rep_list[idx] = self.history[idx][userId][rep_type]
        return rep_list

    def get_word_rep_list(self, word: str, period: np.ndarray) -> Dict[int, np.ndarray]:
        rep_list = dict()
        word_rep = self.words_rep[word]

        for idx in period:
            rep_list[idx] = word_rep
        return rep_list

    def get_df_within_period(self, period: np.ndarray) -> pd.core.frame.DataFrame:
        df_within_period = self.df.loc[period]
        return df_within_period

    def get_tweet_df(self, tweetId: int) -> pd.core.frame.DataFrame:
        tweet_df = self.df[self.df['tweetId'] == tweetId]
        return tweet_df

    def get_user_df(self, userId: str, start_dt=DUMMY_DATE, end_dt=DUMMY_DATE) -> pd.core.frame.DataFrame:
        if start_dt == DUMMY_DATE and end_dt == DUMMY_DATE:
            user_df = self.df[self.df['userId'] == userId]
        else:
            df = self.df.loc[self.get_period_by_date(start_dt=start_dt, end_dt=end_dt)]
            user_df = df[df['userId'] == userId]
        return user_df

    def calc_cosine_similarities(self, rep_lists=List[Dict[int, np.ndarray]]) -> Dict[int, np.ndarray]:
        period = list(rep_lists[0].keys())
        n_rep_lists = len(rep_lists)
        cos_sims = dict()

        for idx in period:
            cos_sims[idx] = np.zeros(n_rep_lists - 1)
            for j, rep_list in enumerate(rep_lists[1:]):
                cos_sims[idx][j] = calc_similarity(rep_1=rep_lists[0][idx], rep_2=rep_list[idx])

        return cos_sims

    def split_cos_sims_dict(self, cos_sims: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        timestamps = self.get_timestamp_by_period(np.array(list(cos_sims.keys())))
        similarities = [[] for _ in range(len(list(cos_sims.values())[0]))]

        for sims in cos_sims.values():
            for i, sim in enumerate(sims):
                similarities[i].append(sim)

        return timestamps, similarities
