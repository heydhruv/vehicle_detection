from base import db
from base.com.vo.register_vo import RegisterVO

class RegisterDAO:
    def insert_register(self, register_vo):
        db.session.add(register_vo)
        db.session.commit()