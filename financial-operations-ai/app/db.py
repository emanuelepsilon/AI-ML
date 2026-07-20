from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import database_url


class Base(DeclarativeBase):
    pass


def _engine(url):
    options = {"future": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            options["poolclass"] = StaticPool
    return create_engine(url, **options)


engine = _engine(database_url())
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_database(url):
    global engine, SessionLocal
    engine.dispose()
    engine = _engine(url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db():
    from app import models  # noqa: F401

    Base.metadata.create_all(engine)


def reset_db():
    from app import models  # noqa: F401

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
