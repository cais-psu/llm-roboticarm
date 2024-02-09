import utils

class RoboticArmFunctions:
    def __init__(self, init_list):
        self.configs = self.load_all_resource_data(init_list)
        self.loggers = {}

    @staticmethod
    def load_all_resource_data(init_list):
        all_configs = {}
        for init in init_list:
            config_data = utils.load_json_data(init)
            all_configs.update(config_data)
        return all_configs

    def sorting(self, product_name: str) -> str:
        """
        This is a sorting function. When executed the product is sorted.
        The function returns whether the operation was successful.

        :param product_name: name of the product to be sorted
        """
        print(f"Sorting product {product_name}")
        return "Success"
    
    def assembly(self, product_name: str) -> str:
        """
        This is an assembly function. When executed the product is assemblied.
        The function returns whether the operation was successful.

        :param product_name: name of the product to be assemblied
        """
        print(f"Assemblying product {product_name}")
        return "Success"