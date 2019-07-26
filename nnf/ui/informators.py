

class TrainingInformator(object):

    def __init__(self):
        self.total_errors = []

    def add_total_error(self, error):
        self.total_errors.append(error)

    def print_total_errors(self):
        pass


informator = TrainingInformator()
