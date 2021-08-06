from redis_connector import RedisConnector


class InferenceRunner:
    def __init__(self):
        self.__redis = RedisConnector()
        