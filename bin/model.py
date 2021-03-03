from pathlib import Path
from typing import List, Union, Tuple, Dict
from tqdm.notebook import tqdm as tqdm_nb

import numpy as np
from gensim.models import word2vec

import myutil
from dataset import Holomem, Hololive


prj_path = myutil.PROJECT_PATH


def log(msg: Union[str, Path], exception=False) -> str:
    suffix = '[{}] '.format(Path(__name__))
    if isinstance(msg, Path):
        msg = str(msg)

    if exception:
        Exception(suffix + msg)
    else:
        print(suffix + msg)


def load_params_file(params_path: Path) -> Tuple[
        Dict[str, Union[int, float]], Dict[str, Union[str, float, int]]]:
    """パラメータファイルから各モデルのパラメータ設定を読み込む

    Args:
        params_path (Path): パラメータファイルのパス

    Returns:
        Tuple[ Dict[str, Union[int, float]], Dict[str, Union[str, float, int]]]: \
            Word2VecModel, ProposalModelのパラメータ設定
    """
    w2v_params, prop_params = {}, {}

    # パラメータファイルの各行は"モデル名 パラメータ名 パラメータ値"という形式になっている
    with open(params_path, mode='r') as f:
        line = f.readline().split(' ')
        model_name, param_name, param_value = line[0], line[1], line[2]

        if model_name == 'w2v':
            # Word2VecModelパラメータ読み込み
            if param_name in {'window', 'size', 'sample', 'negative', 'min_count', 'epochs'}:
                if param_name == 'sample':
                    w2v_params[param_name] = float(param_value)
                else:
                    w2v_params[param_name] = int(param_value)
            else:
                log('指定したWord2VecModelのパラメータ"{}"は未定義です'.format(param_name))
        elif model_name == 'prop':
            # ProposalModelパラメータ読み込み
            if param_name in {'tweet_construct_type', 'user_construct_type', 'tweet_lambda', 'user_context'}:
                if param_name == 'tweet_lambda':
                    prop_params[param_name] = float(param_value)
                elif param_name == 'user_context':
                    prop_params[param_name] = int(param_value)
                else:
                    prop_params[param_name] = param_value
            else:
                log('指定したProposalModelのパラメータ"{}"は未定義です'.format(param_name))
        else:
            log('指定したモデル"{}"は未定義です'.format(model_name))

    return w2v_params, prop_params


def calc_similarity(rep_1: np.ndarray, rep_2: np.ndarray) -> float:
    """ベクトル間のコサイン類似度を計算する

    Args:
        rep_1 (np.ndarray): ベクトル1
        rep_2 (np.ndarray): ベクトル2（ベクトル1とベクトル2が逆でも計算結果は同じ）

    Returns:
        float: ベクトル1とベクトル2のコサイン類似度
    """
    cos_sim = float(np.dot(rep_1, rep_2) / (np.linalg.norm(rep_1) * np.linalg.norm(rep_2)))
    return cos_sim


class Word2VecModel:
    """Word2Vec(gensim)モデル
    """

    def __init__(self, hololive: Hololive, model_name: str, verbose=False):
        """インスタンス化

        Args:
            hololive (Hololive): Hololiveクラス（データセット）
            model_name (str): モデル名
            verbose (bool, optional): 保存パスなどの情報出力. Defaults to False.
        """
        # モデルファイル名 -> ${Hololive.fname}_${model_name}
        self.model_path = Path(prj_path, 'model/word2vec/{0}_{1}.model'.format(hololive.fname, model_name))
        self.loaded_model = False   # 既にモデルが読み込まれているか（もしくは訓練済みか）
        self.sentences_path = hololive.wakatigaki_path  # センテンスファイル（訓練データ）のパス
        self.verbose = verbose

        # モデルファイルが既に存在 -> インスタンス生成時にモデルとパラメータを読み込む
        if self.model_path.exists():
            self.model = self.__load_model()    # モデル読み込み
            self.params = self.__load_params()  # パラメータ読み込み
            self.loaded_model = True

    def __load_model(self) -> word2vec.Word2VecModel:
        """既存モデルファイルを読み込む

        Returns:
            word2vec.Word2VecModel: Word2Vecモデルファイル
        """
        model = word2vec.Word2Vec.load(str(self.model_path))

        if self.verbose:
            log('以下パスからWord2Vecモデルを読み込みました')
            log(self.model_path)
        return model

    def __load_params(self) -> Dict[str, Union[int, float]]:
        """モデルのハイパーパラメータを読み込む

        Returns:
            Dict[str, Union[int, float]]: ハイパーパラメータ設定
        """
        model = self.model  # 先に__load_modelでモデルを読み込んでおく

        # チューニング対象の6つのハイパーパラメータをまとめておく
        params = dict(
            window=model.window, size=model.vector_size, sample=model.sample,
            negative=model.negative, min_count=model.min_count, epochs=model.epochs
        )
        return params

    def fit(self, params: Dict[str, Union[int, float]], save=True) -> None:
        """モデルをセンテンスファイルと指定したパラメータ設定により訓練する

        Args:
            params (Dict[str, Union[int, float]]): Word2Vecモデルのハイパーパラメータ設定
            save (bool, optional): モデルのmodel形式での保存. Defaults to True.
        """

        # モデル読み込みor訓練済み -> 例外を投げて再訓練させない
        if self.loaded_model:
            log('モデルは既に読み込みor訓練済みです', exception=True)

        self.params = params    # ハイパーパラメータ設定
        # 分かち書き結果ファイルをセンテンスファイルとして読み込み
        sentences = word2vec.LineSentence(str(self.sentences_path))

        # モデル訓練
        model = word2vec.Word2Vec(sentences=sentences, window=params['window'], size=params['size'],
                                  sample=params['sample'], negative=params['negative'], min_count=params['min_count'],
                                  epochs=params['epochs'], sg=1, seed=923, workers=1)

        if save:
            model.save(str(self.model_path))
            if self.verbose:
                log('以下パスにWord2Vecモデルを保存しました')
                log(self.model_path)

        self.loaded_model = True
        self.model = model

    def show_params(self) -> None:
        """チューニング対象のハイパーパラメータ設定を表示する
        """
        log('Word2Vec ハイパーパラメータ')
        for name, value in self.params.items():
            log(name, value)

    def get_word_rep(self, word: str) -> np.ndarray:
        """学習された単語のベクトルを取得する

        Args:
            word (str): 単語

        Returns:
            np.ndarray: 単語ベクトル
        """
        vector = self.model.wv[word]
        return vector

    def get_similar_words(self, word: str, topn: int) -> List[Tuple[str, float]]:
        """指定した単語と類似する単語を取得する

        Args:
            word (str): 単語
            topn (int): トップ取得数

        Returns:
            List[Tuple[str, float]]: wordと類似する（単語ベクトル間のコサイン類似度が高い）単語ベストtopn
        """
        similar_words = self.model.most_similar(positive=word, topn=topn)
        return similar_words

    def calc_similarity(self, word_1: str, word_2: str) -> float:
        """単語ベクトル間のコサイン類似度を計算する

        Args:
            word_1 (str): 単語1
            word_2 (str): 単語2（単語1と単語2が逆でも計算結果は同じ）

        Returns:
            float: 単語1と単語2の対応する単語ベクトル間のコサイン類似度
        """
        cos_sim = self.model.wv.similarity(w1=word_1, w2=word_2)
        return cos_sim


class ProposalModel:
    """提案モデル
    """

    def __init__(self, hololive: Hololive, w2v_model: Word2VecModel, params: Dict[str, Union[str, float, int]]):
        """インスタンス化

        Args:
            hololive (Hololive): Hololive（データセット）
            w2v_model (Word2VecModel): Word2Vecモデル（読み込みor訓練済み）
            params (Dict[str, Union[str, float, int]]): 提案モデルのハイパーパラメータ設定
        """
        # 読み込みor訓練済みのWord2Vecモデルのみ受け付ける
        if not w2v_model.loaded_model:
            log('与えたWord2Vecモデルは未訓練状態です', exception=True)
        self.w2v_model = w2v_model

        self.train = hololive.train_session_df  # 訓練セット（セッションデータ形式）
        self.test = hololive.test_session_df    # テストセット（セッションデータ形式）
        self.params = params
        self.userIds = hololive.userIds

        # データセットに含まれるユーザ全員分，ProposalUserクラス（提案モデル用のユーザクラス）のインスタンスを生成
        self.users = dict()
        for uId in self.userIds:
            self.users[uId] = ProposalUser(userId=uId, n_tweet=hololive.n_tweet,
                                           w2v_model=self.w2v_model, load=True)

    def __get_isLast(self, isTrain: bool) -> Dict[int, bool]:
        """ツイートの末尾単語のフラグリストを取得する

        Args:
            isTrain (bool): Trueで訓練セット，Falseでテストセットを指定

        Returns:
            Dict[int, bool]: ツイートの末尾単語のログがTrueになるフラグリスト \
                isLastはユーザ表現構築の合図として使用する
        """
        # セッションデータ準備
        if isTrain:
            session_data = self.train
        else:
            session_data = self.test

        tIds = session_data['tweetId'].values   # ツイートIDリスト
        indices = session_data.index.to_numpy   # ログインデックスリスト
        isLast = dict()  # フラグリスト
        # 単語数2未満のツイートはデータセットの整形段階で除外しているため，最初のログのフラグは必ずFalse
        isLast[indices[0]] = False
        prv_tId = tIds[0]   # 直前のツイートIDを保持

        for idx, tId in zip(indices[1:], tIds[1:]):
            # 1つのツイートログの系列内に他のツイートログが混ざることはない
            if tId != prv_tId:
                # 直前のツイートIDから変化 -> 直前のログの単語がそのツイートIDの末尾単語
                isLast[idx - 1] = True
                prv_tId = tId
            else:
                isLast[idx] = False

        isLast[-1] = True   # 1つのツイートがデータセットを跨ぐことはないため，最後のログの単語は必ずツイート末尾単語
        return isLast

    def learn(self, isTrain=True) -> None:
        if isTrain:
            # 訓練セットによる学習 -> ユーザ表現集合をリセット
            self.user_reps = dict()
            session_data = self.train
        else:
            # テストセットによる学習
            if self.user_reps is None:
                # ユーザ表現集合がリセット状態 -> 先に訓練セットを使って学習を行うよう例外を投げる
                log('テストセットを適用する前に訓練セットを使ってモデルの学習を行ってください', exception=True)
            else:
                # テストセットでは各表現をログ単位で記録する（ヒストリー）
                session_data = self.test
                self.reps_history = dict()

                # 訓練セットによる学習が終了した段階での各ユーザ表現を記録しておく
                start_idx = session_data.index.to_numpy()[0] - 1
                for uId in self.userIds:
                    self.reps_history[start_idx][uId]['userId'] = self.user_reps[uId].user_rep

        isLast = self.__get_isLast(train=isTrain)   # ツイートの末尾単語のフラグリスト取得
        for tLog in tqdm_nb(session_data.itertuples(), total=session_data.shape[0]):
            idx, tId, uId, word = tLog.Index, tLog.tweetId, tLog.userId, tLog.word

            # 単語分散表現が取得できない（訓練セットに存在しない単語） -> 各表現の構築・更新は行わない
            try:
                self.w2v_model.get_word_rep(word)
            except KeyError:
                continue

            # ユーザuIdのツイート・ユーザ表現の構築・更新
            self.users[uId].update_reps(idx=idx, tId=tId, word=word, isLast=isLast[idx])

            # テストセット -> ヒストリー記録
            if not isTrain:
                # ユーザuId以外の各表現は変化しないためコピー
                self.reps_history[idx] = self.reps_history[idx - 1].copy()

                # ユーザuIdのツイート表現は単語分散表現が取得できれば必ず構築・更新されるため記録
                self.reps_history[idx][uId]['tweetId'] = self.user_reps[uId].session_rep
                # ツイート末尾単語のフラグが立っている -> ユーザ表現が必ず構築・更新されるため記録
                if isLast[idx]:
                    self.reps_history[idx][uId]['userId'] = self.user_reps[uId].user_rep


class ProposalUser(Holomem):
    def __init__(self, userId: str, n_tweet: int, proposal_model: ProposalModel,
                 load=True, verbose=False, preview=False):
        super().__init__(userId, n_tweet, load=load, verbose=verbose, preview=preview)
        self.reps_history = dict()
        self.tweetIds = [-1]
        self.tweet_reps = dict()
        self.user_rep = None

        self.proposal_model = proposal_model
        self.w2v_model = proposal_model.w2v_model
        self.tweet_rep_config = proposal_model.tweet_rep_config
        self.user_rep_config = proposal_model.user_rep_config

    @property
    def tweet_rep(self):
        return self.tweet_reps[-1]

    def update_reps(self, idx: int, tId: int, word: str, isLast: bool) -> None:
        self.__construct_session_rep(idx=idx, tId=tId, word=word)
        if (self.tweet_reps[tId] is not None) and isLast:
            self.__construct_user_rep(idx=idx, tId=tId)

    def __get_word_rep(self, word: str) -> Union[np.ndarray, bool]:
        try:
            word_rep = self.w2v_model.get_word_rep(word=word)
        except KeyError:
            word_rep = False
        return word_rep

    def __construct_session_rep(self, idx: int, tId: int, word: str) -> bool:
        latest_tId = self.sessionIds[-1]
        word_rep = self.__get_word_rep(word=word)

        if word_rep:
            return False

        if tId != latest_tId:
            self.sessionIds.append(tId)
            self.tweet_reps[tId] = word_rep
        else:
            session_rep_construct_type = self.proposal_model.rep_construct_types['session']
            tweet_rep = self.session_reps[latest_tId].copy()

            if session_rep_construct_type == 'cos':
                cos_sim = calc_similarity(rep_1=tweet_rep, rep_2=word_rep)
                weight = abs(cos_sim)
                updated_tweet_rep = weight * tweet_rep + (1 - weight) * word_rep
            elif session_rep_construct_type == 'odd':
                lmb = self.tweet_rep_config['lmb']
                weight = np.exp(-lmb)
                updated_tweet_rep = weight * tweet_rep + word_rep
            else:
                log('指定したセッション表現構築タイプ"{}"は未定義です', exception=True)

            self.session_reps[latest_tId] = updated_tweet_rep

    def __construct_user_rep(self):
        latest_tId = self.tweetIds[-1]
        latest_tweet_rep = self.tweet_reps[latest_tId]

        context_size = self.user_rep_config['context']
        if context_size < 1:
            log('コンテキストサイズ（指定値: {}）は1以上の整数を指定してください', exception=True)
        past_tIds = self.tweetIds[-context_size:]

        weighted_rep_sum = np.zeros(self.w2v_model.params['size'])
        sum_weight = 0.0

        for tId in past_tIds:
            past_tweet_rep = self.tweet_reps[tId]
            cos_sim = calc_similarity(rep_1=latest_tweet_rep, rep_2=past_tweet_rep)
            weight = abs(cos_sim)

            weighted_rep_sum += weight * past_tweet_rep
            sum_weight += weight

        updated_user_rep = weighted_rep_sum / sum_weight
        self.user_rep = updated_user_rep