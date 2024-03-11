from server.db.models.qa_info import QaInfoModel
from server.db.mysql_session import with_session


@with_session
def add_qainfo_to_db(session, work_code, work_name, question, answer, prompt, kb, context, sources, id):
    # 创建知识库实例
    # work_code = Column(String, comment='')
    # name = Column(String, comment='')
    # question = Column(String, comment="")
    # answer = Column(String, default="")
    # prompt = Column(String, default="")
    # kb = Column(String, default="")
    kb = QaInfoModel(id=id, work_code=work_code, work_name=work_name,
                     question=question, answer=answer, prompt=prompt,
                     kb=kb, context=context, sources=sources)
    session.add(kb)
    return kb.id
@with_session
def feed_back(
    session, id, thumb, feedback
):
    session.query(QaInfoModel).filter(QaInfoModel.id == id).update({"thumb": thumb, "feedback": feedback})

@with_session
def list_qainfo_from_db(session, min_file_count: int = -1):
    kbs = session.query(QaInfoModel.kb_name).filter(QaInfoModel.file_count > min_file_count).all()
    kbs = [kb[0] for kb in kbs]
    return kbs


@with_session
def kb_exists(session, kb_name):
    kb = session.query(QaInfoModel).filter_by(kb_name=kb_name).first()
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    kb = session.query(QaInfoModel).filter_by(kb_name=kb_name).first()
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_session
def delete_kb_from_db(session, kb_name):
    kb = session.query(QaInfoModel).filter_by(kb_name=kb_name).first()
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    kb: QaInfoModel = session.query(QaInfoModel).filter_by(kb_name=kb_name).first()
    if kb:
        return {
            "kb_name": kb.kb_name,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}
