import sqlite3

class Database(object):
    def __init__(self, config):
        self.config = config
        db_path = os.path.join(config['db_path'], config['db_name'])
        self.conn = sqlite3.connect(db_path+'.db')

    def getCursor(self):
        pass

    def executeQuery(self):
        pass
