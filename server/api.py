import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from server.chat.utils import History
from configs.model_config import chat_list
from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
import uuid as myuuid
from fastapi import Body, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, JSONResponse
from server.chat.chat import chat
from server.chat.search_engine_chat import search_engine_chat
from server.chat.completion import completion
from server.chat.feedback import chat_feedback
from server.embeddings_api import embed_texts_endpoint
from server.llm_api import (
    list_running_models,
    list_config_models,
    change_llm_model,
    stop_llm_model,
    get_model_config,
    list_search_engines,
)
from server.utils import (
    BaseResponse,
    ListResponse,
    FastAPI,
    MakeFastAPIOffline,
    get_server_configs,
    get_prompt_template,
)
from typing import List, Literal

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

from server.db.repository.qa_info_repository import feed_back


async def document():
    return RedirectResponse(url="/docs")


async def index(request: Request):
    if check_mobile(request):
        return RedirectResponse(url="/mobile.html")
    else:
        return RedirectResponse(url="/index.html")


def check_mobile(request):
    user_agent = request.headers.get("User-Agent", "")
    viewtype = request.query_params.get("viewtype")
    if viewtype == "mobile":
        return True
    return "Mobile" in user_agent or "DingTalk" in user_agent


def request_sse(
    query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
    knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
    history: List[History] = Body(
        [],
        description="历史对话",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
):
    uuid = myuuid.uuid1()
    chat_list[str(uuid)] = {
        "query": query,
        "knowledge_base_name": knowledge_base_name,
        "history": history,
    }
    return JSONResponse(content={"data": str(uuid)})


def chat_feed_back(
    id: str = Body(..., description="ID"),
    thumb: str = Body(..., description="1：赞，0：踩", examples=[1]),
    feedback: str = Body(..., description="反馈内容", examples=["回答不准确"]),
):
    feed_back(id, thumb, feedback)
    return JSONResponse(content={"data": str("OK")})


def create_app(run_mode: str = None):
    app = FastAPI(title="Langchain-Chatchat API Server", version=VERSION)
    MakeFastAPIOffline(app)
    origins = ["http://localhost:8083"]
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/", response_model=BaseResponse, summary="swagger 文档")(index)

    # Tag: Chat
    app.post(
        "/chat/chat",
        tags=["Chat"],
        summary="与llm模型对话(通过LLMChain)",
    )(chat)

    from server.chat.chat import chat_n

    app.get("/chat/chatn/{uid}", tags=["Chat"], summary="与llm模型对话(通过LLMChain)")(chat_n)

    app.post(
        "/chat/search_engine_chat",
        tags=["Chat"],
        summary="与搜索引擎对话",
    )(search_engine_chat)

    app.post(
        "/chat/feedback",
        tags=["Chat"],
        summary="返回llm模型对话评分",
    )(chat_feedback)

    # 知识库相关接口
    mount_knowledge_routes(app)
    # 摘要相关接口
    mount_filename_summary_routes(app)

    # LLM模型相关接口
    app.post(
        "/llm_model/list_running_models",
        tags=["LLM Model Management"],
        summary="列出当前已加载的模型",
    )(list_running_models)

    app.post(
        "/llm_model/list_config_models",
        tags=["LLM Model Management"],
        summary="列出configs已配置的模型",
    )(list_config_models)

    app.post(
        "/llm_model/get_model_config",
        tags=["LLM Model Management"],
        summary="获取模型配置（合并后）",
    )(get_model_config)

    app.post(
        "/llm_model/stop",
        tags=["LLM Model Management"],
        summary="停止指定的LLM模型（Model Worker)",
    )(stop_llm_model)

    app.post(
        "/llm_model/change",
        tags=["LLM Model Management"],
        summary="切换指定的LLM模型（Model Worker)",
    )(change_llm_model)

    # 服务器相关接口
    app.post(
        "/server/configs",
        tags=["Server State"],
        summary="获取服务器原始配置信息",
    )(get_server_configs)

    app.post(
        "/server/list_search_engines",
        tags=["Server State"],
        summary="获取服务器支持的搜索引擎",
    )(list_search_engines)

    @app.post(
        "/server/get_prompt_template",
        tags=["Server State"],
        summary="获取服务区配置的 prompt 模板",
    )
    def get_server_prompt_template(
        type: Literal[
            "llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"
        ] = Body(
            "llm_chat",
            description="模板类型，可选值：llm_chat，knowledge_base_chat，search_engine_chat，agent_chat",
        ),
        name: str = Body("default", description="模板名称"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # 其它接口
    app.post(
        "/other/completion",
        tags=["Other"],
        summary="要求llm模型补全(通过LLMChain)",
    )(completion)

    app.post(
        "/other/embed_texts",
        tags=["Other"],
        summary="将文本向量化，支持本地模型和在线模型",
    )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    from server.chat.knowledge_base_chat import knowledge_base_chat
    from server.chat.file_chat import upload_temp_docs, file_chat
    from server.chat.agent_chat import agent_chat
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from server.knowledge_base.kb_doc_api import (
        list_files,
        upload_docs,
        delete_docs,
        update_docs,
        download_doc,
        recreate_vector_store,
        search_docs,
        DocumentWithVSId,
        update_info,
        update_docs_by_id,
    )

    app.post(
        "/chat/search_engine_chat",
        tags=["Chat"],
        summary="与搜索引擎对话",
    )(search_engine_chat)

    app.post(
        "/chat/feedback",
        tags=["Chat"],
        summary="返回llm模型对话评分",
    )(chat_feedback)

    # 知识库相关接口
    mount_knowledge_routes(app)
    # 摘要相关接口
    mount_filename_summary_routes(app)

    # LLM模型相关接口
    app.post(
        "/llm_model/list_running_models",
        tags=["LLM Model Management"],
        summary="列出当前已加载的模型",
    )(list_running_models)

    app.post(
        "/llm_model/list_config_models",
        tags=["LLM Model Management"],
        summary="列出configs已配置的模型",
    )(list_config_models)

    app.post(
        "/llm_model/get_model_config",
        tags=["LLM Model Management"],
        summary="获取模型配置（合并后）",
    )(get_model_config)

    app.post(
        "/llm_model/stop",
        tags=["LLM Model Management"],
        summary="停止指定的LLM模型（Model Worker)",
    )(stop_llm_model)

    app.post(
        "/llm_model/change",
        tags=["LLM Model Management"],
        summary="切换指定的LLM模型（Model Worker)",
    )(change_llm_model)

    # 服务器相关接口
    app.post(
        "/server/configs",
        tags=["Server State"],
        summary="获取服务器原始配置信息",
    )(get_server_configs)

    app.post(
        "/server/list_search_engines",
        tags=["Server State"],
        summary="获取服务器支持的搜索引擎",
    )(list_search_engines)

    @app.post(
        "/server/get_prompt_template",
        tags=["Server State"],
        summary="获取服务区配置的 prompt 模板",
    )
    def get_server_prompt_template(
        type: Literal[
            "llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"
        ] = Body(
            "llm_chat",
            description="模板类型，可选值：llm_chat，knowledge_base_chat，search_engine_chat，agent_chat",
        ),
        name: str = Body("default", description="模板名称"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # 其它接口
    app.post(
        "/other/completion",
        tags=["Other"],
        summary="要求llm模型补全(通过LLMChain)",
    )(completion)

    app.post(
        "/other/embed_texts",
        tags=["Other"],
        summary="将文本向量化，支持本地模型和在线模型",
    )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    from server.chat.knowledge_base_chat import knowledge_base_chat
    from server.chat.knowledge_base_chat import knowledge_base_chatn
    from server.chat.file_chat import upload_temp_docs, file_chat
    from server.chat.agent_chat import agent_chat
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from server.knowledge_base.kb_doc_api import (
        list_files,
        upload_docs,
        delete_docs,
        update_docs,
        download_doc,
        recreate_vector_store,
        search_docs,
        DocumentWithVSId,
        update_info,
        update_docs_by_id,
    )

    app.post("/chat/knowledge_base_chat", tags=["Chat"], summary="与知识库对话")(
        knowledge_base_chat
    )

    app.get("/chat/knowledge_base_chatn/{uid}", tags=["Chat"], summary="与知识库对话")(
        knowledge_base_chatn
    )

    app.post("/request/sse", response_model=BaseResponse, summary="请求")(request_sse)

    app.post("/chat/search_engine_chat", tags=["Chat"], summary="与搜索引擎对话")(
        search_engine_chat
    )
    app.post("/chat/file_chat", tags=["Knowledge Base Management"], summary="文件对话")(
        file_chat
    )

    app.post("/chat/agent_chat", tags=["Chat"], summary="与agent对话")(agent_chat)

    app.post("/chat/chat_feedback", tags=["Chat"], summary="反馈")(chat_feed_back)

    # Tag: Knowledge Base Management
    app.get(
        "/knowledge_base/list_knowledge_bases",
        tags=["Knowledge Base Management"],
        response_model=ListResponse,
        summary="获取知识库列表",
    )(list_kbs)

    app.post(
        "/knowledge_base/create_knowledge_base",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="创建知识库",
    )(create_kb)

    app.post(
        "/knowledge_base/delete_knowledge_base",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="删除知识库",
    )(delete_kb)

    app.get(
        "/knowledge_base/list_files",
        tags=["Knowledge Base Management"],
        response_model=ListResponse,
        summary="获取知识库内的文件列表",
    )(list_files)

    app.post(
        "/knowledge_base/search_docs",
        tags=["Knowledge Base Management"],
        response_model=List[DocumentWithVSId],
        summary="搜索知识库",
    )(search_docs)

    app.post(
        "/knowledge_base/update_docs_by_id",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="直接更新知识库文档",
    )(update_docs_by_id)

    app.post(
        "/knowledge_base/upload_docs",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="上传文件到知识库，并/或进行向量化",
    )(upload_docs)

    app.post(
        "/knowledge_base/delete_docs",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="删除知识库内指定文件",
    )(delete_docs)

    app.post(
        "/knowledge_base/update_info",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="更新知识库介绍",
    )(update_info)
    app.post(
        "/knowledge_base/update_docs",
        tags=["Knowledge Base Management"],
        response_model=BaseResponse,
        summary="更新现有文件到知识库",
    )(update_docs)

    app.get(
        "/knowledge_base/download_doc",
        tags=["Knowledge Base Management"],
        summary="下载对应的知识文件",
    )(download_doc)

    app.post(
        "/knowledge_base/recreate_vector_store",
        tags=["Knowledge Base Management"],
        summary="根据content中文档重建向量库，流式输出处理进度。",
    )(recreate_vector_store)

    app.post(
        "/knowledge_base/upload_temp_docs",
        tags=["Knowledge Base Management"],
        summary="上传文件到临时目录，用于文件对话。",
    )(upload_temp_docs)


def mount_filename_summary_routes(app: FastAPI):
    from server.knowledge_base.kb_summary_api import (
        summary_file_to_vector_store,
        recreate_summary_vector_store,
        summary_doc_ids_to_vector_store,
    )

    app.post(
        "/knowledge_base/kb_summary_api/summary_file_to_vector_store",
        tags=["Knowledge kb_summary_api Management"],
        summary="单个知识库根据文件名称摘要",
    )(summary_file_to_vector_store)
    app.post(
        "/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
        tags=["Knowledge kb_summary_api Management"],
        summary="单个知识库根据doc_ids摘要",
        response_model=BaseResponse,
    )(summary_doc_ids_to_vector_store)
    app.post(
        "/knowledge_base/kb_summary_api/recreate_summary_vector_store",
        tags=["Knowledge kb_summary_api Management"],
        summary="重建单个知识库文件摘要",
    )(recreate_summary_vector_store)


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="langchain-ChatGLM",
        description="About langchain-ChatGLM, local knowledge based ChatGLM with langchain"
        " ｜ 基于本地知识库的 ChatGLM 问答",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
