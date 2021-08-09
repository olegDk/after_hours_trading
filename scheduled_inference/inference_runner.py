from redis_connector import RedisConnector
from rabbit_sender import RabbitSender
import time
import traceback

RABBIT_MQ_HOST = 'rabbit'
RABBIT_MQ_PORT = 5672
ORDER_RELATED_DATA = 'orderRelatedData'


class InferenceRunner:
    def __init__(self):
        self.__redis = RedisConnector()
        self.__rabbit = RabbitSender(RABBIT_MQ_HOST, RABBIT_MQ_PORT)

    def run_scheduler(self):
        try:
            while True:
                print('Run some inference work')
                print('Pulling data from Redis')
                l1_dict = self.__redis.get_dict('l1')
                print(l1_dict)
                print('Sending test order log')
                test_message = [{'messageId': 'test',
                                'data': ['test_data']}]
                self.__rabbit.send_message(message=test_message,
                                           routing_key=ORDER_RELATED_DATA)
                time.sleep(10)
        except Exception as e:
            print(f'Inside run_scheduler Inference runner'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            self.run_scheduler()
