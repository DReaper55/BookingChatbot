import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.products.accessory_retrieval_service import AccessoryRetrievalService
from src.services.products.book_retrieval_service import BookRetrievalService
from src.services.products.dress_retrieval_service import DressRetrievalService
from src.services.products.electronics_retrieval_service import ElectronicsRetrievalService
from src.services.products.food_retrieval_service import FoodRetrievalService
from src.services.products.furniture_retrieval_service import FurnitureRetrievalService
from src.services.products.jacket_retrieval_service import JacketRetrievalService
from src.services.products.pants_retrieval_service import PantsRetrievalService
from src.services.products.shirt_retrieval_service import ShirtRetrievalService
from src.services.bookings.train_retrieval_service import TrainRetrievalService
from src.services.products.shoe_retrieval_service import ShoeRetrievalService
from src.services.products.toy_retrieval_service import ToyRetrievalService
from src.utils.singleton_meta import SingletonMeta


class RetrievalService(metaclass=SingletonMeta):
    """Service class responsible for retrieving different types of products."""

    # .....................................
    # Train Retrieval
    # .....................................
    @staticmethod
    def find_train(train_day, train_departure, train_destination):
        return TrainRetrievalService.find_train(train_day, train_departure, train_destination)

    @staticmethod
    def buy_train(train_id, passenger_name):
        return TrainRetrievalService.buy_train(train_id, passenger_name)

    # .....................................
    # Food Retrieval
    # .....................................
    @staticmethod
    def find_food(food_item_type, restaurant_name):
        return FoodRetrievalService.find_food(food_item_type, restaurant_name)

    @staticmethod
    def buy_food(food_id, quantity):
        return FoodRetrievalService.buy_food(food_id, quantity)

    # .....................................
    # Shirt Retrieval
    # .....................................
    @staticmethod
    def find_shirt(**kwargs):
        return ShirtRetrievalService.find_shirt(**kwargs)

    @staticmethod
    def buy_shirt(**kwargs):
        return ShirtRetrievalService.buy_shirt(**kwargs)

    # .....................................
    # Dress Retrieval
    # .....................................
    @staticmethod
    def find_dress(**kwargs):
        return DressRetrievalService.find_dress(**kwargs)

    @staticmethod
    def buy_dress(**kwargs):
        return DressRetrievalService.buy_dress(**kwargs)

    # .....................................
    # Pants Retrieval
    # .....................................
    @staticmethod
    def find_pants(**kwargs):
        return PantsRetrievalService.find_pants(**kwargs)

    @staticmethod
    def buy_pants(**kwargs):
        return PantsRetrievalService.buy_pants(**kwargs)

    # .....................................
    # Jacket Retrieval
    # .....................................
    @staticmethod
    def find_jacket(**kwargs):
        return JacketRetrievalService.find_jacket(**kwargs)

    @staticmethod
    def buy_jacket(**kwargs):
        return JacketRetrievalService.buy_jacket(**kwargs)

    # .....................................
    # Book Retrieval
    # .....................................
    @staticmethod
    def find_book(**kwargs):
        return BookRetrievalService.find_book(**kwargs)

    @staticmethod
    def buy_book(**kwargs):
        return BookRetrievalService.buy_book(**kwargs)

    # .....................................
    # Furniture Retrieval
    # .....................................
    @staticmethod
    def find_furniture(**kwargs):
        return FurnitureRetrievalService.find_furniture(**kwargs)

    @staticmethod
    def buy_furniture(**kwargs):
        return FurnitureRetrievalService.buy_furniture(**kwargs)

    # .....................................
    # Electronics Retrieval
    # .....................................
    @staticmethod
    def find_electronics(**kwargs):
        return ElectronicsRetrievalService.find_electronics(**kwargs)

    @staticmethod
    def buy_electronics(**kwargs):
        return ElectronicsRetrievalService.buy_electronics(**kwargs)

    # .....................................
    # Accessory Retrieval
    # .....................................
    @staticmethod
    def find_accessory(**kwargs):
        return AccessoryRetrievalService.find_accessory(**kwargs)

    @staticmethod
    def buy_accessory(**kwargs):
        return AccessoryRetrievalService.buy_accessory(**kwargs)

    # .....................................
    # Toy Retrieval
    # .....................................
    @staticmethod
    def find_toy(**kwargs):
        return ToyRetrievalService.find_toy(**kwargs)

    @staticmethod
    def buy_toy(**kwargs):
        return ToyRetrievalService.buy_toy(**kwargs)

    # .....................................
    # Shoe Retrieval
    # .....................................
    @staticmethod
    def find_shoe(**kwargs):
        return ShoeRetrievalService.find_shoe(**kwargs)

    @staticmethod
    def buy_shoe(**kwargs):
        return ShoeRetrievalService.buy_shoe(**kwargs)
