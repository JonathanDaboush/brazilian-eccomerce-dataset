import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


BACKEND_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BACKEND_DIR / "runtime"
RUNTIME_DIR.mkdir(exist_ok=True)


def build_database_url() -> str:
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_database = os.getenv("MYSQL_DATABASE")

    if mysql_user and mysql_password and mysql_database:
        mysql_host = os.getenv("MYSQL_HOST", "mysql")
        mysql_port = os.getenv("MYSQL_PORT", "3306")
        return (
            f"mysql+pymysql://{mysql_user}:{mysql_password}"
            f"@{mysql_host}:{mysql_port}/{mysql_database}"
        )

    sqlite_path = RUNTIME_DIR / "ecommerce_demo.db"
    return f"sqlite:///{sqlite_path}"


DATABASE_URL = build_database_url()
SQLITE_CONNECT_ARGS = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=SQLITE_CONNECT_ARGS,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


class ReplayBatch(Base):
    __tablename__ = "replay_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(128), default="olist_order_events")
    transport: Mapped[str] = mapped_column(String(32), default="direct")
    topic: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    requested_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pace_ms: Mapped[int] = mapped_column(Integer, default=0)
    total_events: Mapped[int] = mapped_column(Integer, default=0)
    processed_events: Mapped[int] = mapped_column(Integer, default=0)
    failed_events: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    events: Mapped[list["ProcessedEvent"]] = relationship(back_populates="batch")


class ProcessedEvent(Base):
    __tablename__ = "processed_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("replay_batches.id"), nullable=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    event_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="processed")
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    batch: Mapped[ReplayBatch | None] = relationship(back_populates="events")


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_unique_id: Mapped[str | None] = mapped_column(String(64), index=True)
    customer_zip_code_prefix: Mapped[int | None] = mapped_column(Integer, nullable=True)
    customer_city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    customer_state: Mapped[str | None] = mapped_column(String(8), nullable=True)


class Seller(Base):
    __tablename__ = "sellers"

    seller_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seller_zip_code_prefix: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seller_city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seller_state: Mapped[str | None] = mapped_column(String(8), nullable=True)


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_category_name: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    product_name_lenght: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_description_lenght: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_photos_qty: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_weight_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_length_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    product_width_cm: Mapped[float | None] = mapped_column(Float, nullable=True)


class CategoryTranslation(Base):
    __tablename__ = "category_translation"

    product_category_name: Mapped[str] = mapped_column(String(128), primary_key=True)
    product_category_name_english: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Geolocation(Base):
    __tablename__ = "geolocation"

    geolocation_zip_code_prefix: Mapped[int] = mapped_column(Integer, primary_key=True)
    geolocation_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    geolocation_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    geolocation_city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    geolocation_state: Mapped[str | None] = mapped_column(String(8), nullable=True)


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.customer_id"), nullable=True, index=True)
    order_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    order_purchase_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    order_approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    order_delivered_carrier_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    order_delivered_customer_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    order_estimated_delivery_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    customer: Mapped[Customer | None] = relationship()


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), index=True)
    order_item_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    product_id: Mapped[str | None] = mapped_column(ForeignKey("products.product_id"), nullable=True, index=True)
    seller_id: Mapped[str | None] = mapped_column(ForeignKey("sellers.seller_id"), nullable=True, index=True)
    shipping_limit_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    freight_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class Payment(Base):
    __tablename__ = "payments"

    payment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), index=True)
    payment_sequential: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payment_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payment_installments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payment_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class Review(Base):
    __tablename__ = "reviews"

    review_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    review_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.order_id"), index=True)
    review_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_comment_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_comment_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_creation_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    review_answer_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def reset_demo_db() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
