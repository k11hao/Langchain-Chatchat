from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker

from configs import SQLALCHEMY_DATABASE_URI
import json
from configs import db_qa

engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()


mysql_engine = create_engine(
    'mysql+pymysql://'+db_qa['user']+':'+db_qa['pwd']+'@'+db_qa['host']+':3306/'+db_qa['dbname']+'?charset=utf8',
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

MysqlSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=mysql_engine)