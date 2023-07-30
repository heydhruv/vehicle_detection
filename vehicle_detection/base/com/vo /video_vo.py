from base import db

class VideoVO(db.Model):
    __tablename__ = 'video_table'
    video_id = db.Column('video_id', db.Integer, primary_key=True,
                           autoincrement=True)
    video_date = db.Column('video_date', db.DATETIME, nullable=False)
    input_video = db.Column('input_video', db.String(255),
                                    nullable=False)
    # video_status = db.Column('video_status', db.String(255),
    #                                nullable=False)
    output_video = db.Column('output_video', db.String(255),
                                   nullable=False)
    video_entry_count = db.Column('video_entry_count', db.Integer,
                                  nullable=False)
    video_exit_count = db.Column('video_exit_count', db.Integer,
                                 nullable=False)

    def as_dict(self):
        return {
            'video_id': self.video_id,
            'video_date': self.video_date,
            'input_video': self.input_video,
            'output_video': self.output_video,
            'video_entry_count': self.video_entry_count,
            'video_exit_count': self.video_exit_count
        }


db.create_all()