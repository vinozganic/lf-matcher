import logging
import os
import sys
import json

from time import sleep

from pika import BlockingConnection, PlainCredentials, ConnectionParameters, URLParameters

from contracts import item_to_process
from data_transformer import DataTransformer
from matcher import Matcher
from model import Model

class MatcherService:
    def __init__(self, connection_parameters, matcher):
        self.connection_parameters = connection_parameters
        self.matcher = matcher
        self.connection = None
        self.channel = None
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def start(self):
        while not self.connection:
            try:
                self.connection = BlockingConnection(self.connection_parameters)
                self.channel = self.connection.channel()
                logging.info("Connected to RabbitMQ.")
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
    connection_parameters = None
    if os.environ.get("RABBITMQ_HOST") is None:
        amqp_endpoint = os.environ["AMQP_ENDPOINT"]
        connection_parameters = URLParameters(amqp_endpoint)
    else:
        host = os.environ["RABBITMQ_HOST"]
        port = os.environ["RABBITMQ_PORT"]
        username = os.environ["RABBITMQ_USERNAME"]
        password = os.environ["RABBITMQ_PASSWORD"]
        credentials = PlainCredentials(username, password)
        connection_parameters = ConnectionParameters(host, port, '/', credentials)
    model = Model()
    model.load_model(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model/model.pkl")
    data_transformer = DataTransformer()
    matcher = Matcher(model=model, data_transformer=data_transformer)
    config = {}
    
    with MatcherService(connection_parameters=connection_parameters, matcher=matcher) as service:
        service.start()