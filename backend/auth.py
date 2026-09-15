from sqlalchemy.orm import Session
from passlib.context import CryptContext

from models import User


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def register_user(
    db: Session,
    name: str,
    email: str,
    password: str
):

    existing_user = db.query(User).filter(
        User.email == email
    ).first()

    if existing_user:
        return None

    hashed_password = pwd_context.hash(
        password
    )

    new_user = User(
        name=name,
        email=email,
        password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


def login_user(
    db: Session,
    email: str,
    password: str
):

    user = db.query(User).filter(
        User.email == email
    ).first()

    if user is None:
        return None

    if not pwd_context.verify(
        password,
        user.password
    ):
        return None

    return user