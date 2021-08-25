from trader import Trader
import time
import traceback
from redis_connector import RedisConnector

RABBIT_MQ_HOST = 'rabbit'
RABBIT_MQ_PORT = 5672
ORDER_RELATED_DATA = 'orderRelatedData'


class InferenceRunner:
    def __init__(self):
        self.__redis = RedisConnector()
        self.__trader = Trader()

    def run_scheduler(self):
        try:
            while True:
                print('Try to run some inference work')
                print('Pulling data from Redis')
                l1_dict = self.__redis.get_dict('l1')
                if l1_dict:
                    print('l1_dict not empty, running inference')
                    orders = self.__trader.run_inference(l1_dict)
                    # # Sending orders logic
                    # print('Sending test order log')
                    # test_message = [{'messageId': 'test',
                    #                 'data': ['test_data']}]
                    # self.__trader.send_order_log_to_mq(log=test_message)
                    print("Sending orders logic")
                time.sleep(10)
        except Exception as e:
            print(f'Inside run_scheduler inference_runner'
                  f'An exception of type {type(e).__name__}. Arguments: '
                  f'{e.args}')
            print(traceback.format_exc())
            self.run_scheduler()


# runner = InferenceRunner()
