from redis_connector import RedisConnector
import time


class InferenceRunner:
    def __init__(self):
        self.__redis = RedisConnector()

    # def run_scheduler(self):
    #     try:
    #         while True:
    #             print('Run some inference work')
    #             print('Pulling data from Redis')
    #             l1_
