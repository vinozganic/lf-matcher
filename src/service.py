import logging
import os
import sys
import json

from time import sleep

from pika import BlockingConnection, URLParameters

from contracts import item_to_process
from data_transformer import DataTransformer
from matcher import Matcher
from model import Model

class MatcherService:
    def __init__(self, amqp_endpoint, matcher):
        self.amqp_endpoint = amqp_endpoint
        self.matcher = matcher
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def start(self):
        connected = False
        while not connected:
            try:
                self.connection = BlockingConnection(URLParameters(self.amqp_endpoint))
                self.channel = self.connection.channel()
                connected = True
            except Exception as e:
                logging.info("Error connecting to AMQP endpoint. Retrying in 5 seconds...")
                sleep(5)

        self.channel.queue_declare(queue='matcher.item_to_process', durable=True)
        self.channel.basic_consume(queue='matcher.item_to_process', on_message_callback=self.on_message_callback, auto_ack=True)
        self.channel.start_consuming()


    def on_message_callback(self, ch, method, properties, body):
        decoded_message = json.loads(body)
        logging.info("Received message %r" % decoded_message)
        to_process = item_to_process.from_dict(decoded_message)
        self.matcher.process_message(to_process)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.channel.stop_consuming()
        self.connection.close()

if __name__ == "__main__":
    amqp_endpoint = os.environ["AMQP_ENDPOINT"]
    model = Model()
    model.load_model(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model/model.pkl")
    data_transformer = DataTransformer()
    matcher = Matcher(model=model, data_transformer=data_transformer)
    config = {}
    
    with MatcherService(amqp_endpoint=amqp_endpoint, matcher=matcher) as service:
        service.start()