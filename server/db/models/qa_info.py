from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func

from server.db.base import Base

class QaInfoModel(Base):
    """
    问答记录
    """
    __tablename__ = 'qa_his_test'
    id = Column(String, primary_key=True, comment='ID')
    work_code = Column(String, comment='')
    work_name = Column(String, comment='')
    question = Column(String, comment="")
    answer = Column(String, default="")
    prompt = Column(JSON, default=[])
    kb = Column(String, default="")
    context = Column(String, default="")
    sources = Column(JSON, default=[])
    feedback = Column(String, default='')
    thumb = Column(Integer, default=0)
    create_time = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<qa_his(id='{self.id}', work_code='{self.work_code}', question='{self.question}', answer='{self.answer}', prompt='{self.prompt}')>"
