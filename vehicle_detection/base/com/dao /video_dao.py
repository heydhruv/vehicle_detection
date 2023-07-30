from base import db
from base.com.vo.video_vo import VideoVO


class VideoDAO:
    def insert_video(self, video_vo):
        db.session.add(video_vo)
        db.session.commit()

    def view_video(self):
        video_vo_list = VideoVO.query.all()
        return video_vo_list

    def delete_video(self, video_id):
        video_vo_list = VideoVO.query.get(video_id)
        db.session.delete(video_vo_list)
        db.session.commit()
        return video_vo_list

    def total_count(self):
        video_vo_list = VideoVO.query.all()
        return video_vo_list