import pika
from pika.exceptions import AMQPConnectionError
import socket
import sys
import os
import json
import time
from pytz import timezone
from datetime import datetime
from typing import Tuple
import traceback
from config.constants import *
from news_analyzer.news_analyzer import NewsAnalyzer
from redis_connector import RedisConnector

EST = timezone('EST')
now_init = datetime.now()
na = NewsAnalyzer()
rc = RedisConnector(REDIS_EXT_HOST, REDIS_PORT)
weekday = datetime.today().weekday()


def process_news(news_data: dict):
    try:
        if news_data:
            content = news_data.get(CONTENT)
            dt = datetime.strptime(news_data.get(DATETIME),
                                   DATETIME_FORMAT)
            dt_hour = dt.hour
            dt_aligned = datetime(year=dt.year,
                                  month=dt.month,
                                  day=dt.day,
                                  hour=now_init.hour,
                                  minute=now_init.minute,
                                  second=now_init.second)
            # Filter by datetime
            delta_days = (now_init - dt_aligned).days
            valid_delta_days = 1 if not weekday == 0 else 3
            postmarket_condition = delta_days == valid_delta_days and dt_hour >= 16  # Automate monday detection
            premarket_condition = delta_days == 0
            if premarket_condition or postmarket_condition:
                relevant = na.is_relevant(text=content)
                symbol = news_data[SYMBOL]
                news_data[NEWS_RELEVANCE_KEY] = relevant
                symbol_news_data = rc.hm_get(h=NEWS_TYPE, key=symbol)[0]
                if symbol_news_data:
                    print(f'Got symbol_news_data for symbol {symbol}: '
                          f'{symbol_news_data}')
                    symbol_news_data = json.loads(symbol_news_data)
                    symbol_news_data[symbol][N_NEWS] += 1
                    # News type is news heading not topic
                    symbol_news_data[symbol][NEWS_TYPE] = \
                        symbol_news_data[symbol][NEWS_TYPE] + [news_data]
                    symbol_news_data_str = json.dumps(symbol_news_data)
                    rc.h_set_str(h=NEWS_TYPE,
                                 key=symbol,
                                 value=symbol_news_data_str)
                else:
                    print('Got news for first time!')
                    symbol_news_data = {
                        symbol: {
                            N_NEWS: 1,
                            NEWS_TYPE: [news_data]
                        }
                    }
                    symbol_news_data_str = json.dumps(symbol_news_data)
                    rc.h_set_str(h=NEWS_TYPE,
                                 key=symbol,
                                 value=symbol_news_data_str)
                if relevant:
                    rc.h_set_float(h=STOCK_TO_TIER_PROPORTION,
                                   key=symbol, value=0)
    except Exception as e:
        message = f'Process news: ' \
                  f'An exception of type {type(e).__name__} occurred. Arguments:{e.args}'
        print(message)
        print(traceback.format_exc())
        print(f'Failed to process news list: {news_data}')
        pass


def connect_rabbit() -> Tuple[pika.BlockingConnection,
                              pika.adapters.blocking_connection.BlockingChannel]:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to rabbit from receive "
                  f"news...")
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBIT_MQ_EXT_HOST,
                                          port=RABBIT_MQ_PORT))
            channel = connection.channel()

            channel.queue_declare(queue=NEWS_TYPE)
            print(f"Connected to rabbit from receive "
                  f"news successfully")
            print(f"========================================================="
                  f"===================================================")
            return connection, channel
        except socket.gaierror:
            print(f"Failed connecting to rabbit from receive "
                  f"news, maybe it didn't start yet "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except ConnectionError:
            print(f"Failed connecting to rabbit from receive "
                  f"news, ConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)
        except AMQPConnectionError:
            print(f"Failed connecting to rabbit from receive "
                  f"news, AMQPConnectionError "
                  f"sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


def main():
    connection, channel = connect_rabbit()

    def callback(ch, method, properties, body):
        message = body.decode('UTF-8')
        news_data = json.loads(message)[0]
        print(f"\n\n [x] Received news")
        print(news_data)
        process_news(news_data=news_data)

    channel.basic_consume(queue=NEWS_TYPE,
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
