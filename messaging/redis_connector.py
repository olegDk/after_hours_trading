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
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                            decode_responses=True)
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

    def set_dict(self, name: str, d: dict, rewrite: bool = False):
        try:
            if rewrite:
                self.h_del(h=name)
            self.__redis.hmset(name, d)
        except Exception as e:
            print(f'Inside set_dict RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            print(f'Failed to set dict: {d}')

    def hm_get(self, h: str, key: str) -> str:
        try:
            value = self.__redis.hmget(h, key)
            # print(f'Got value in hm_get for hash: {h}, '
            #       f'and key: {key}: {value}')
            return value
        except Exception as e:
            print(f'Inside hm_get RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            return ''

    def h_set_int(self, h: str, key: str, value: int):
        try:
            # print(f'Setting value: {value} for key: {key}, '
            #       f'in hash: {h}')
            self.__redis.hset(name=h,
                              key=key,
                              value=value)
        except Exception as e:
            print(f'Inside h_set_float RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())

    def h_set_float(self, h: str, key: str, value: float):
        try:
            # print(f'Setting value: {value} for key: {key}, '
            #       f'in hash: {h}')
            self.__redis.hset(name=h,
                              key=key,
                              value=value)
        except Exception as e:
            print(f'Inside h_set_float RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())

    def h_set_str(self, h: str, key: str, value: str):
        try:
            # print(f'Setting value: {value} for key: {key}, '
            #       f'in hash: {h}')
            self.__redis.hset(name=h,
                              key=key,
                              value=value)
        except Exception as e:
            print(f'Inside h_set_str RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())

    def h_getall(self, h: str) -> dict:
        try:
            value = self.__redis.hgetall(h)
            # print(f'Got value in h_getall for hash {h}: {value}')
            return value
        except Exception as e:
            print(f'Inside h_getall RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            return {}

    def h_del(self, h: str):
        try:
            all_keys = list(self.__redis.hgetall(h).keys())
            if all_keys:
                self.__redis.hdel(h, *all_keys)
        except Exception as e:
            print(f'Inside h_del RedisConnector'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
