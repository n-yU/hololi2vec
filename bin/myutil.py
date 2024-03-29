import os
from typing import Tuple, Union
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parents[1]


def log(msg: Union[str, Path], exception=False) -> str:
    suffix = '[{}] '.format(Path(__name__))
    if isinstance(msg, Path):
        msg = str(msg)

    if exception:
        raise Exception(suffix + msg)
    else:
        print(suffix + msg)


def load_twitter_api_keys() -> Tuple[str, str, str, str]:
    """APIキーをまとめているファイルから各キーを読み込む

    Returns:
        Tuple[str, str, str, str]: api key, api key secret, access token, access token secret
    """
    # キーファイルを読み込んでリストにまとめる
    keys_file_path = Path(PROJECT_PATH, 'twitter.key')
    with open(keys_file_path, mode='r', encoding='utf-8') as f:
        keys_file = f.readlines()
    keys_file = [line.rstrip(os.linesep) for line in keys_file]

    # 各キーを対応する変数に代入
    keys_dict = dict()
    for key in keys_file:
        name, value = key.split(':')
        keys_dict[name] = value

    api_key = keys_dict['api key']
    api_secret = keys_dict['api key secret']
    access_token = keys_dict['access token']
    access_secret = keys_dict['access token secret']

    return api_key, api_secret, access_token, access_secret
