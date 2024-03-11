import json, requests


def query_user(token: str):
    res = UcenterResponse()
    if token is None or token == '':
        res.code = 500
        data = json.dumps({"answer": "非法访问"}, ensure_ascii=False)
        res.data = data
        return res

    url = 'https://appcenter.tqls.cn/ucenter/api/v1/users/info'

    res = requests.get(url=url, headers={"Content-Type": "application/json", "token": token})
    if res.status_code != 200:
        data = json.dumps({"answer": "无法查到用户信息"}, ensure_ascii=False)
        res.code = res.status_code
        res.data = data
        return res, None

    user_response = res.text

    user_info_json = json.loads(user_response)

    if user_info_json['code'] != 200:
        data = json.dumps({"answer": "获取您的个人信息失败，可能登录过期，请重新登录"}, ensure_ascii=False)
        res.code = user_info_json['code']
        res.data = data
        return res, None
    print(user_info_json)
    sex = '男'
    if user_info_json['data']['user']['sex'] == 'female':
        sex = '女'


    #'，职位：' + user_info_json['data']['user']['positionName'] + \
    data = '姓名：' + user_info_json['data']['name'] + '，工号：' + user_info_json['data']['user'][
        'loginName'] + '，性别：' + sex + \
        '，职务：' + \
        user_info_json['data']['user']['dutyName']
    res.code = 200
    res.data = data
    return res, user_info_json['data']


class UcenterResponse:
    code: int
    msg: str
    data: str
