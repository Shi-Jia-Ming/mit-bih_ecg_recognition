import MySQLdb as sql

"""
数据库信息: 
- host: 数据库ip地址
- port: 数据库端口(默认为3306)
- user: 数据库账户用户名
- passwd: 数据库账户密码
- db: 数据库名称
"""


class DbInterface:
    # 初始化数据库
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

        self._connect = sql.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            db=self.db,
        )

        self._cursor = self._connect.cursor()

    # 读取数据库数据：床垫获得的最新的数据
    def read_intelligent_mattress(self):
        # 查询数据库中的最新一条数据
        read_sql = "select * from intelligent_mattress order by id desc limit 1"
        self._cursor.execute(read_sql)
        rst = self._cursor.fetchone()
        # print(rst[11])  # 心率原始数据
        return eval(rst[11])
