from base import app
from flask import Flask, render_template, request, redirect

from werkzeug.utils import secure_filename
import os
from base.com.dao.video_dao import VideoDAO
from base.com.vo.video_vo import VideoVO
from datetime import datetime
from base.services.multi_updown_obj_det_and_trk import detect
from base.services import multi_updown_obj_det_and_trk
from base.com.controller.login_controller import admin_login_session, admin_logout_session


INPUT_FOLDER = 'base/static/adminResources/video'
app.config['INPUT_FOLDER'] = INPUT_FOLDER

OUTPUT_FOLDER = 'base/static/adminResources/output_video'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/home_load')
def home():
    try :
        if admin_login_session() == "admin":
            return render_template('admin/index.html')
        else:
            return admin_logout_session()
    except Exception as ex:
        return str(ex)

@app.route('/upload_video')
def upload_video():
    try:
        if admin_login_session() == "admin":
            return render_template('admin/add_video.html')
        else:
            return admin_logout_session()
    except Exception as ex:
        return str(ex)


@app.route('/add_video', methods=['POST'])
def add_video():
    try :
        if admin_login_session() == "admin":
            input_video = request.files.get("video")
            video_name = secure_filename(input_video.filename)
            # print(video_name)
            #input_video_path = os.path.join(app.config['VIDEO_FOLDER'])
            input_video_path = os.path.join(app.config['INPUT_FOLDER'])

            input_video.save(os.path.join(input_video_path, video_name))

            print(">>>>>>>>>>>>>>>>>")
            detect_video = input_video_path + "/" + video_name
            output_path, entry_count, exit_count = detect(source=detect_video)


            # output_video = request.files.get("output_video")
            # #output_video_path = os.path.join(app.config['OUTPUT_FOLDER'])
            # output_video_path =app.config['OUTPUT_FOLDER']
            # output_video.save(os.path.join(output_video_path, video_name))

            video_dao = VideoDAO()
            video_vo = VideoVO()

            current_date = datetime.now()
            print(">>>>>",current_date)
            video_vo.video_date = current_date
            video_vo.input_video = detect_video.replace("base","..")
            video_vo.output_video = output_path.replace("base","..")
            video_vo.video_entry_count = entry_count
            video_vo.video_exit_count = exit_count

            video_dao.insert_video(video_vo)




            # input={"video_input":input_video_path}
            # input_list.append(input)
            #
            # output={"video_output":output_path}
            # out_list.append(output)


            return render_template('admin/add_video.html')
        else:
            return admin_logout_session()
    except Exception as ex:
        return str(ex)

@app.route('/view_history')
def view_history():
    try :

        if admin_login_session() == "admin":
            video_dao = VideoDAO()
            video_vo_list = video_dao.view_video()
            return render_template('admin/view_history.html', video_vo_list = video_vo_list)
        else:
            return admin_logout_session()

    except Exception as ex:
        return str(ex)

@app.route('/delete_video')
def delete_video():
    try:
        if admin_login_session() == "admin":
            video_dao = VideoDAO()
            video_vo = VideoVO()
            video_id = request.args.get('videoId')
            video_vo.video_id = video_id
            video_vo_list = video_dao.delete_video(video_id)
            file_path = video_vo_list.input_video.replace("..","base")
            output_file_path=video_vo_list.output_video.replace("..","base")
            os.remove(file_path)
            os.remove(output_file_path)
            return redirect("/view_history")
        else:
            return admin_logout_session()

    except Exception as ex:
        return str(ex)

@app.route('/about')
def about():
    try:
        if admin_login_session() == "admin":
            return render_template('admin/about.html')
        else:
            return admin_logout_session()
    except Exception as ex:
        return str(ex)