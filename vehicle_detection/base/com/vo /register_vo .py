from base import db
from base.com.vo.login_vo import LoginVO

class RegisterVO(db.Model):
    __tablename__ = 'register_table'
    __table_args__ = {'extend_existing': True}
    register_id = db.Column('register_id', db.Integer, primary_key=True,
                         autoincrement=True)
    user_name = db.Column('user_name', db.String(255), nullable=False)
    email = db.Column('email', db.String(255), nullable=False)
    country = db.Column('country', db.String(255), nullable=False)
    password = db.Column('password', db.String(255), nullable=False)
    login_id = db.Column('login_id', db.Integer,
                         db.ForeignKey(LoginVO.login_id,
                                       ondelete='CASCADE',
                                       onupdate='CASCADE'),
                         nullable=False)
    def as_dict(self):
        return {
            'register_id': self.register_id,
            'user_name': self.user_name,
            'email': self.email,
            'country': self.country,
            'password': self.password,
            'login_id': self.login_id

        }


db.create_all()