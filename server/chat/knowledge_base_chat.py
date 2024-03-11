import uuid

from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs import (Default_LLM_MODEL,LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, ONLINE_LLM_MODEL,
                     chat_list)
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
import json
import os
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
# from server.chat.user_service import query_user

import zhipuai

from server.db.repository.qa_info_repository import (
   add_qainfo_to_db
)


from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0, le=1),
                              history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                 "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                 "content": "虎头虎脑"}]]
                                                            ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(Default_LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              prompt_name: str = Body("knowledge_base_chat",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                              request: Request = None,
                              ):
    print("--knowledge_base_name:"+str(knowledge_base_name))
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]
    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        # nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=[callback],
        )
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        docs = await run_in_threadpool(search_docs,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold)

        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name":filename})
            url = f"/knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token,
                                  "docs": source_documents},
                                 ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer
            # yield json.dumps({"answer": answer,
            #                   "docs": source_documents},
            #                  ensure_ascii=False)

        await task
    print("model_name-------6:",model_name)
    return StreamingResponse(knowledge_base_chat_iterator(query, kb, top_k, history, model_name),
                             media_type="text/event-stream")


async def knowledge_base_chatn(uid: str, request: Request):
    print("----knowledge_base_chatn-------")

    if uid not in chat_list:
        return BaseResponse(code=500, msg="非法访问")

    param = chat_list[uid]

    query = param['query']
    history = param['history']
    model_name = ''
    if 'model_name' in param and param['model_name'] is not None:
        model_name = param['model_name']

    if model_name == '':
        model_name = Default_LLM_MODEL

    knowledge_base_name = param['knowledge_base_name']
    stream = True
    local_doc_url = False
    top_k = VECTOR_SEARCH_TOP_K
    score_threshold = SCORE_THRESHOLD

    del chat_list[uid]

    # knowledge_base_chat(query=param['query'],
    #                     history=param['history'],
    #                     knowledge_base_name=param['knowledge_base_name'],
    #                     stream=True,
    #                     local_doc_url=False,
    #                     top_k=VECTOR_SEARCH_TOP_K,
    #                     request=request)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History(**h) if isinstance(h, dict) else h for h in history]

    async def knowledge_base_chat_iterator_glm_online(query: str,
                                                      kb: str,
                                                      top_k: int,
                                                      history: Optional[List[History]],
                                                      ) -> AsyncIterable[str]:

        zhipuai.api_key = ONLINE_LLM_MODEL[Default_LLM_MODEL]["api_key"]
        prompts = []
        # if len(history) == 0:
        #     history.append('')

        # for item in history:
        #     if len(item) > 0:
        #         prompts.append({"role": "user", "content": item[0]})
        #         prompts.append({"role": "bot", "content": item[1]})

        prompt_template = get_prompt_template("knowledge_base_chat", knowledge_base_name)

        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])

        user_token = request.cookies.get('token', None)
        if user_token is None:
            user_token = request.query_params.get('token')
        if user_token is None:
            user_token = request.headers.get('token')
        # u_response, user_info_json = query_user(user_token)
        # if u_response.code != 200:
        #     yield f"data: {u_response.data}\n\n"
        #     yield "event: done\ndata: \n\n"

        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        # .replace("userinfo",u_response.data)

        # if len(history) == 0:
        #     prompts.append({"role": "bot", "content": context})

        # prompts.append({"role": "user", "content": PROMPT_TEMPLATE_MAP[knowledge_base_name]['first_question']})
        # prompts.append({"role": "assistant", "content": context})
        prompts.append({"role": "user", "content": prompt})

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        # bot_info = PROMPT_TEMPLATE_MAP[knowledge_base_name]['bot_info'].replace("{context}", context)
        response = zhipuai.model_api.sse_invoke(
            model=ONLINE_LLM_MODEL[Default_LLM_MODEL]['name'],
            # meta={
            #     "user_info": u_response.data,
            #     "bot_info": bot_info,
            #     "bot_name": PROMPT_TEMPLATE_MAP[knowledge_base_name]['bot_name'],
            #     "user_name": user_info_json['name']
            # },
            prompt=prompts,
            temperature=0.95,
            top_p=0.7,
            incremental=True
        )

        ans_res = ''
        yield f"event: start\ndata:  \n\n"

        try:
            for event in response.events():
                if event.event == "add":
                    ans_res += event.data
                    data = json.dumps({"answer": event.data}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
                elif event.event == "error" or event.event == "interrupted":
                    yield f"event: error\ndata: {event.data}\n\n"
                    print("server-error", event.data)
                elif event.event == "finish":
                    print("finish")
                    print(event.meta)
                else:
                    print(event)

            # data = json.dumps({"answer": "", "docs": source_documents}, ensure_ascii=False)
            # yield f"data: {data}\n\n"
            yield f"event: done\ndata: {data}\n\n"
            try:
                add_qainfo_to_db(work_code=user_info_json['user']['loginName'],
                             work_name=user_info_json['user']['trueName'],
                             question=query,
                             answer=ans_res,
                             prompt=prompts,
                             context=context,
                             sources=source_documents,
                             kb=knowledge_base_name)
            except Exception as e:
                print(e)

            # history[-1] = [prompt, ans_res]
        except Exception as e:
            print(e)

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           kb: str,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODELS
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            callbacks=[callback],
        )

        # Begin a task that runs in the background.
        user_token = request.cookies.get('token', None)
        if user_token is None:
            user_token = request.query_params.get('token')
        # u_response, user_info_json = query_user(user_token)

        # if u_response.code != 200:
        #     yield f"event: start\ndata:  \n\n"

        #     data = json.dumps({"answer": u_response.data, "docs": ""}, ensure_ascii=False)
        #     yield f"data: {data}\n\n"
        #     yield f"event: done\ndata:  \n\n"
        #     return


        ##为了查询更准确搜索时，把用户信息放进去
        doc_query = query; # + "; " + u_response.data
        docs = search_docs(doc_query, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])

        # prompt_template = get_prompt_template('knowledge_base_chat')
        prompt_template = get_prompt_template("knowledge_base_chat", kb)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            # chain.acall({"context": context, "question": query, "userinfo": u_response.data}),
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        ans_res = ''
        yield f"event: start\ndata:  \n\n"
        kb_id = str(uuid.uuid1())
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                data = json.dumps({"answer": token}, ensure_ascii=False)
                ans_res += data
                yield f"data: {data}\n\n"
            data = json.dumps({"answer": "", "docs": source_documents, "id": kb_id}, ensure_ascii=False)
            # yield f"data: {data}\n\n"
            yield f"event: done\ndata: {data}\n\n"
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer, "docs": source_documents}, ensure_ascii=False)


        try:
            add_qainfo_to_db(id=kb_id, work_code=user_info_json['user']['loginName'],
                         work_name=user_info_json['user']['trueName'],
                         question=query,
                         answer=ans_res,
                         prompt=prompt_template,
                         context=context,
                         sources=source_documents,
                         kb=knowledge_base_name)
        except Exception as e:
            print(e)
        await task

    if Default_LLM_MODEL.startswith('chatglm-online'):
        return StreamingResponse(knowledge_base_chat_iterator_glm_online(query, knowledge_base_name, top_k, history),
                                 media_type="text/event-stream")
    else:
        return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                              top_k=top_k,
                                                              history=history,
                                                              kb=knowledge_base_name,
                                                              model_name=model_name),
                                 media_type="text/event-stream")
    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))

