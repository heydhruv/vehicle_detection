from flask import render_template, request, redirect, flash,session,make_response,url_for

from base import app
from base.com.dao.register_dao import RegisterDAO
from base.com.vo.register_vo import RegisterVO

from base.com.vo.login_vo import LoginVO
from base.com.dao.login_dao import LoginDAO

from base.com.dao.video_dao import VideoDAO
from base.com.vo.video_vo import VideoVO

from datetime import timedelta

import bcrypt


global_loginvo_list = []
global_login_secretkey_set = {0}

@app.route('/')
def login():
    try:

        return render_template('admin/login.html')
    except Exception as ex:
        return str(ex)


@app.route('/validate_login', methods=['POST','GET'])
def load_login():
    try:
        global global_loginvo_list
        global global_login_secretkey_set

        login_username = request.form.get("username")
        login_password = request.form.get("password").encode("utf-8")



        login_vo = LoginVO()
        login_dao = LoginDAO()

        login_vo.user_name = login_username

        login_vo_list = login_dao.check_login_username(login_vo)
        login_list = [i.as_dict() for i in login_vo_list]
        len_login_list = len(login_list)


        if len_login_list == 0:
            error_message = 'Username or password is incorrect!!'
            flash(error_message)
            return redirect("/")
        else:
            login_id = login_list[0]['login_id']
            login_username = login_list[0]['user_name']
            login_role = login_list[0]['login_role']
            login_secretkey = login_list[0]['login_secretkey']
            hashed_login_password = login_list[0]['password'].encode("utf-8")
            print(hashed_login_password)

            if bcrypt.checkpw(login_password, hashed_login_password):
                login_vo_dict = {
                    login_secretkey: {'login_username': login_username,
                                      'login_role': login_role,
                                      'login_id': login_id}}
                if len(global_loginvo_list) != 0:
                    for i in global_loginvo_list:
                        temp_list = list(i.keys())
                        global_login_secretkey_set.add(temp_list[0])
                    login_secretkey_list = list(global_login_secretkey_set)
                    if login_secretkey not in login_secretkey_list:
                        global_loginvo_list.append(login_vo_dict)
                else:
                    global_loginvo_list.append(login_vo_dict)
                if login_role == 'admin':
                    response = make_response(
                        redirect(url_for('load_dashboard')))
                    response.set_cookie('login_secretkey',
                                        value=login_secretkey,
                                        max_age=timedelta(minutes=30))
                    response.set_cookie('login_username', value=login_username,
                                        max_age=timedelta(minutes=30))
                    return response
                else:
                    return redirect(url_for('admin_logout_session'))
            else:
                error_message = 'username or Password is incorrect!!'
                flash(error_message)
                return redirect("/")

    except Exception as ex:
        return str(ex)

@app.route('/load_forget_password')
def load_forget_password():
    try:
        return render_template('admin/forgetPassword.html')
    except Exception as ex:
        return str(ex)

@app.route('/load_dashboard')
def load_dashboard():
    try:
        video_dao = VideoDAO()

        total_count = video_dao.total_count()
        video_list = [i.as_dict() for i in total_count]

        # total entry count
        total_entry_count = []
        for i in video_list:
            count = int(i.get('video_entry_count'))
            total_entry_count.append(count)
        sum_entry = sum(total_entry_count)

        # total exit count
        total_exit_count = []
        for i in video_list:
            count = int(i.get('video_exit_count'))
            total_exit_count.append(count)
        sum_exit = sum(total_exit_count)
        return render_template('admin/index.html', entry = sum_entry, exit=sum_exit)
    except Exception as ex:
        return str(ex)

@app.route('/admin/login_session')
def admin_login_session():
    try:
        global global_loginvo_list
        login_role_flag = ""
        login_secretkey = request.cookies.get('login_secretkey')
        if login_secretkey is None:
            return redirect('/')
        for i in global_loginvo_list:
            if login_secretkey in i.keys():
                if i[login_secretkey]['login_role'] == 'admin':
                    login_role_flag = "admin"
        return login_role_flag
    except Exception as ex:
        return str(ex)


@app.route("/admin/logout_session", methods=['GET'])
def admin_logout_session():
    try:
        global global_loginvo_list
        login_secretkey = request.cookies.get('login_secretkey')
        login_username = request.cookies.get('login_username')
        response = make_response(redirect('/'))
        if login_secretkey is not None and login_username is not None:
            response.set_cookie('login_secretkey', login_secretkey, max_age=0)
            response.set_cookie('login_username', login_username, max_age=0)
            for i in global_loginvo_list:
                if login_secretkey in i.keys():
                    global_loginvo_list.remove(i)
                    break
        return response
    except Exception as ex:
        return str(ex)

@app.route('/validate_login_username', methods=['POST'])
def validate_login_username():
    try:
        login_username = request.form.get("username")
        login_dao = LoginDAO()
        login_vo = LoginVO()

        login_vo.user_name = login_username
        login_vo_list = login_dao.login_validate_username(login_vo)
        print(">>>>>>>>>>>>>>>>>>>",login_vo_list)
        login_list = [i.as_dict() for i in login_vo_list]
        len_login_list = len(login_list)
        if len_login_list == 0:
            error_message = 'username is incorrect !'
            flash(error_message)
            return redirect(url_for('load_forget_password'))
        else:
            login_id = login_list[0]['login_id']
            session['session_login_id'] = login_id
            # login_username = login_list[0]['login_username']
            return render_template('admin/resetPassword.html')
    except Exception as ex:
        return str(ex)

@app.route('/insert_reset_password', methods=['POST'])
def insert_reset_password():
    try:
        login_password = request.form.get("loginPassword")
        salt = bcrypt.gensalt(rounds=12)
        hashed_login_password = bcrypt.hashpw(login_password.encode("utf-8"),
                                              salt)
        login_id = session.get("session_login_id")
        login_dao = LoginDAO()
        login_vo = LoginVO()
        login_vo.login_id = login_id
        login_vo.password = hashed_login_password
        login_dao.update_login(login_vo)
        return redirect('/')
    except Exception as ex:
        return str(ex)



