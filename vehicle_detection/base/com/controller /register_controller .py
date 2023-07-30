from base import app
from flask import Flask, render_template, request, redirect,flash
from base.com.vo.register_vo import RegisterVO
from base.com.dao.register_dao import RegisterDAO
from base.com.vo.login_vo import LoginVO
from base.com.dao.login_dao import LoginDAO
import random
import bcrypt
import string

login_secretkey=""



@app.route('/load_register')
def load_register():
    try:
        return render_template('admin/register.html')
    except Exception as ex:
        return str(ex)

@app.route('/insert_register', methods=['POST'])
def insert_register():
    try:
        global login_secretkey
        global login_secretkey_flag
        login_secretkey_flag = False

        register_vo = RegisterVO()
        register_dao = RegisterDAO()

        login_vo = LoginVO()
        login_dao = LoginDAO()

        login_username = request.form.get('username')
        login_email = request.form.get('email')
        login_country = request.form.get('country')
        login_password = request.form.get('password')
        login_vo_list = login_dao.view_login()

        login_secretkey_list = [i.as_dict()['login_secretkey'] for i in
                                login_vo_list]
        login_username_list = [i.as_dict()['user_name'] for i in login_vo_list]

        if login_username in login_username_list:
            error_message = "The username is already exists !"
            flash(error_message)
            return redirect('/user/load_user')

        while not login_secretkey_flag:
            login_secretkey = ''.join(
                (random.choice(string.ascii_letters + string.digits)) for x in
                range(32))
            if login_secretkey not in login_secretkey_list:
                login_secretkey_flag = True
                break

        salt = bcrypt.gensalt(rounds=12)
        hashed_login_password = bcrypt.hashpw(login_password.encode("utf-8"),salt)

        login_vo.user_name = login_email
        login_vo.password = hashed_login_password
        login_vo.login_role = "admin"
        login_vo.login_secretkey = login_secretkey
        login_dao.insert_login(login_vo)

        register_vo.user_name = login_username
        register_vo.email = login_email
        register_vo.country = login_country
        register_vo.password = hashed_login_password
        register_vo.login_id = login_vo.login_id
        register_dao.insert_register(register_vo)

        return redirect("/")

    except Exception as ex:
        return str(ex)
