from functools import wraps
from contextlib import contextmanager
from server.db.base import MysqlSessionLocal


@contextmanager
def session_scope():
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = MysqlSessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session:
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except Exception as e:
                session.rollback()
                raise

    return wrapper


def get_db() -> MysqlSessionLocal:
    db = MysqlSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> MysqlSessionLocal:
    db = MysqlSessionLocal()
    return db
