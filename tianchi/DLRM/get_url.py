import requests
import json
import time
import re
import os

fNameExtractor = re.compile(r"131047/(D[0-9]_[0-9]\.csv\.zip)\?Expires")


def get_url(fileId):

    url = f"https://tianchi.aliyun.com/api/dataset/getFileDownloadUrl?fileId={fileId}"

    payload = {}
    headers = {
        "authority": "tianchi.aliyun.com",
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "bx-v": "2.5.1",
        "cookie": "login_current_pk=1599428093307009; _csrf=shhKqtm7ZjlMQ5lNglVDoPdZ; aliyun_lang=zh; tc=s%3AcPu38uzfHVP6EOrKE9SmAEMBt120ITHw.GhyYsNKSzHTKq14w%2BYDVGVhyGkRfvl2L8xKgnLpohG8; _samesite_flag_=true; cookie2=118e95e243b375a5c80bca988dccbf1b; munb=2212353633958; csg=32b2881e; t=c7134b7612106f0c1132273e3e5044ba; _tb_token_=e6ebb5873e3e5; login_aliyunid_ticket=TwUhTOoNC1ZBeeMfKJzxdnb95hYssNIZor6q7SCxRtgmGCbifG2Cd4ZWazmBdHI6sgXZqg4XFWQfyKpeu*0vCmV8s*MT5tJl3_1$$wQvE3tavtRCBfIj5oMp_cBX42tyBNdFFeti5GZbRiLPf_gNpoB_0; login_aliyunid_csrf=_csrf_tk_1370310752309809; login_aliyunid_pk=1599428093307009; hssid=CN-SPLIT-ARCEByIOc2Vzc2lvbl90aWNrZXQyAQE4q8GihuUxQAFKEISZ8tQMIgFVLyvoB0mq4GPhWSdCQUdvOZbBYxU-t6oBg9eeRw; hsite=6; aliyun_country=CN; aliyun_site=CN; login_aliyunid_pks=BG+cCanNtsMCYkhofLX1Fuf1Luodb1jlpDSakOowuGQDvI=; login_aliyunid=%E5%A5%87%E4%B8%96%E4%BF%97%E4%BA%BAa; x-csrf-token=h7dG9zx1-HPf_xq8VnRbPayk4-OH76qCiMaE; aliyun_cert_type_common=1; tfstk=fEdn9nNTFpWC7pqhlAfIsL-f3rD9Jk15x3FR2_IrQGS_weHByg4kvG3CwpeLaLjMVMBdT01lz3LWp7CFLG5NPZreqUgCzg-yqgpKBjLBR_1rD_cxMeso-rOexJyFb47cPJ51AuqJR_1rXPFrHjYBDyuYNwfrS17lzu5PL97ZSwQRaWWzTRrNfa855D2dZSAj-6j0kFfED8M8aiYFjSY6_wPPcjsLa9ROW9Ay8GbOWCbhKiWgn7XkThLy9B6sTWfWWKxkUnoTpMJcohXvKmVHmH_yra-nGrjwtUAGddEULZfhxtRF3bFl_TJMs6dEV-T9SMWc6dHsvTCHxKsfLAi6qFjp4BXaxgW4Q-5Ln7_ZyCy7F971SiKSChah5kVThV0gH86F5ZjvwV2zUoNzwJ0iS-_GLN_c0; isg=BNPTFhn0sgITGH76Kzu6nOwZYlf9iGdK_-AnaIXx9vIpBPSmDFsWmEoSPnRqpL9C",
        "referer": "https://tianchi.aliyun.com/dataset/131047",
        "sec-ch-ua": '"Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.54",
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    return json.loads(response.text)["data"]


for fileId in range(1959219, 1959298 + 1):
    url = get_url(fileId)
    fName = fNameExtractor.findall(url)[0]
    if fName[:4] not in ["D1_0", "D1_1", "D1_2", "D1_3", "D8_0", "D8_1"]:
        time.sleep(5)
        continue
    if (
        os.path.exists(fName) and os.path.getsize(fName) < 2 * 1024 * 1024 * 1024
    ):  # <2GB
        print("Remove", fName, os.path.getsize(fName) // (1024 * 1024), "MB")
        os.remove(fName)
    if not os.path.exists(fName):
        cmd = f'curl -o {fName} "{url}"'
        # print(cmd)
        os.system(cmd)
        os.system("sleep 20")
        # print("sleep 20")
    else:
        print("Skip", fName)
    time.sleep(5)
