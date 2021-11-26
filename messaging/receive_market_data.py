import pika
from pika.exceptions import AMQPConnectionError
import socket
import sys
import os
import json
import time
from pytz import timezone
from typing import Tuple
from datetime import datetime, timedelta
import redis
from config.constants import *

EST = timezone('EST')
L1_KEYS = [PCT_BID_NET, PCT_ASK_NET, BID, ASK,
           BID_VENUE, ASK_VENUE, CLOSE]


def insert_market_data(market_data_list: list, r: redis.Redis):
    if market_data_list:
        l1_dict = r.hgetall(L1)
        now_dt = datetime.now()
        print(f'Current num symbols in redis: {len(l1_dict)}')
        if not l1_dict:
            l1_dict = {}
        for symbol_dict in market_data_list:
            result_dict = {key: symbol_dict[key] for key in L1_KEYS}
            stock_dict = {L1_DATA: result_dict,
                          STOCK_SNAPSHOT: {
                              f'{now_dt.hour}_{now_dt.minute}_{now_dt.second}': {
                                  PCT_BID_NET: result_dict[PCT_BID_NET],
                                  PCT_ASK_NET: result_dict[PCT_ASK_NET]
                              }
                          }}
            l1_dict[symbol_dict[SYMBOL]] = json.dumps(stock_dict)
        r.hmset(L1, l1_dict)
        print(f'Market data inserted')


def connect_redis() -> redis.Redis:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to redis from receive "
                  f"market data...")
            r = redis.Redis(host=REDIS_EXT_HOST, port=REDIS_PORT)
            print(f"Connected to redis from receive "
                  f"market data successfully")
            print(f"========================================================="
                  f"===================================================")
            return r
        except socket.gaierror:
            print(f"Failed connecting to redis from receive "
                  f"market data, maybe it didn't start yet "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except ConnectionError:
            print(f"Failed connecting to redis from receive "
                  f"market data, "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


def connect_rabbit() -> Tuple[pika.BlockingConnection,
                              pika.adapters.blocking_connection.BlockingChannel]:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to rabbit from receive "
                  f"market data...")
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBIT_MQ_EXT_HOST,
                                          port=RABBIT_MQ_PORT))
            channel = connection.channel()

            channel.queue_declare(queue=MARKET_DATA_TYPE)
            print(f"Connected to rabbit from receive "
                  f"market data successfully")
            print(f"========================================================="
                  f"===================================================")
            return connection, channel
        except socket.gaierror:
            print(f"Failed connecting to rabbit from receive "
                  f"market data, maybe it didn't start yet "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except ConnectionError:
            print(f"Failed connecting to rabbit from receive "
                  f"market data, ConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except AMQPConnectionError:
            print(f"Failed connecting to rabbit from receive "
                  f"market data, AMQPConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


def main():
    r = connect_redis()
    connection, channel = connect_rabbit()

    def callback(ch, method, properties, body):
        message = body.decode('UTF-8')
        market_data_list = json.loads(message)
        print(f"\n\n [x] Received market_data list")
        # print(market_data_list)
        insert_market_data(market_data_list=market_data_list, r=r)

    channel.basic_consume(queue=MARKET_DATA_TYPE,
                          on_message_callback=callback,
                          auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


test_dict = {
    'bid': 0,
    'ask': 0,
    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

json.dumps(test_dict)
