import pika
import sys
import os
import json
import time
from pytz import timezone
from datetime import datetime, timedelta
# from pymongo import MongoClient
from elasticsearch import Elasticsearch, ConnectionError, TransportError


EST = timezone('EST')
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
RABBIT_MQ_HOST = 'rabbit'
RABBIT_MQ_PORT = 5672
ORDER_RELATED_DATA = 'orderRelatedData'
ES_HOST = 'elasticsearch'
ES_PORT = 9200
ORDERS_INDEX = 'orders'
ORDERS_SENT_DOC_TYPE = 'ordersSent'


# To save logs in MongoDB Atlas
# client = MongoClient(f"mongodb+srv://olegDk"
#                      f":fy51gl38@cluster0.17cyv.mongodb.net/"
#                      f"myFirstDatabase?retryWrites=true&w=majority")
# db = client.test
# docs = db.docs

# To save logs in Elasticsearch

def es_connect() -> Elasticsearch:
    while True:
        try:
            print(f"========================================================="
                  f"===================================================")
            print(f"Connection attempt to elasticsearch...")
            es = Elasticsearch(hosts=[{"host": ES_HOST, "port": ES_PORT}])
            if not es.indices.exists(index=ORDERS_INDEX):
                es.indices.create(index=ORDERS_INDEX)
            print(f"Connected to elasticsearch successfully")
            print(f"========================================================="
                  f"===================================================")
            return es
        except ConnectionError:
            print(f"Failed connecting to elasticsearch, sleeping for 1 second")
            print(f"========================================================="
                  f"===================================================")
            time.sleep(1)


def insert_orders(orders_list: list, es: Elasticsearch):
    current_time = (datetime.now(EST) +
                    timedelta(hours=1)).strftime(DATE_TIME_FORMAT)
    for order_dict in orders_list:
        order_dict.update({'datetime_est': current_time})
        print(f"\n Received order {order_dict}")
        # To save logs in MongoDB Atlas
        # docs.insert_one(order_dict)

        # To save logs in Elasticsearch
        res = es.index(index=ORDERS_INDEX,
                       doc_type=ORDERS_SENT_DOC_TYPE,
                       body=order_dict)
        print(res)



def main():
    es = es_connect()
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBIT_MQ_HOST,
                                  port=RABBIT_MQ_PORT))
    channel = connection.channel()

    channel.queue_declare(queue=ORDER_RELATED_DATA)

    def callback(ch, method, properties, body):
        message = body.decode('UTF-8')
        orders_list = json.loads(message)
        print(f"\n\n [x] Received orders list")
        insert_orders(orders_list=orders_list, es=es)


    channel.basic_consume(queue=ORDER_RELATED_DATA,
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
