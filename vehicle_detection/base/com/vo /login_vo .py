from base import db

class LoginVO(db.Model):
    __tablename__ = 'login_table'
    __table_args__= {'extend_existing': True}
    login_id = db.Column('login_id', db.Integer, primary_key=True,
                         autoincrement=True)
    user_name = db.Column('user_name', db.String(255), nullable=False)
    password = db.Column('password', db.String(255), nullable=False)
    login_role = db.Column('login_role', db.String(100), nullable=False,
                           default="admin")
    login_secretkey = db.Column('login_secretkey', db.String(100),
                                nullable=False)

    def as_dict(self):
        return {
            'login_id': self.login_id,
            'user_name': self.user_name,
            'password': self.password,
            'login_role': self.login_role,
            'login_secretkey': self.login_secretkey

        }

db.create_all()