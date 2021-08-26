import json

import redis
import socket
import time
import traceback

REDIS_HOST = 'localhost'
REDIS_PORT = 6379


def connect_redis() -> redis.Redis:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to redis from Trader...")
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
            print(f"Connected to redis from Trader successfully...")
            print(f"========================================================="
                  f"===================================================")
            return r
        except socket.gaierror:
            print(f"Failed connecting to redis from Trader, "
                  f"maybe it didn't start yet "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except ConnectionError:
            print(f"Failed connecting to redis from Trader, "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


class RedisConnector:
    def __init__(self):
        self.__redis = connect_redis()

    def get_dict(self, name: str) -> dict:
        try:
            l1_dict = self.__redis.hgetall(name)
            return json.loads(l1_dict)
        except Exception as e:
            print(f'Inside get_dict RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            return {}

    def set_dict(self, name: str, d: dict):
        try:
            print(d)
            print(type(d))
            self.__redis.hmset(name, d)
        except Exception as e:
            print(f'Inside set_dict RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())

    def hm_get(self, h: str, key: str) -> str:
        try:
            value = self.__redis.hmget(h, key)
            print(f'Got value: {value}')
            return value
        except Exception as e:
            print(f'Inside hm_get RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            return ''
