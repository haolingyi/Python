import urllib.request as uReq
import urllib.parse as uPar
import json
content = input('请输入需要翻译的内容:')
url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'

data = {}
data['type'] = 'AUTO'
data['i'] = content
data['doctype'] = 'json'
data['version'] = '2.1'
data['keyfrom'] = 'fanyi.web'
data['typoResult'] = 'ture'
data = uPar.urlencode(data).encode('utf-8')

req = uReq.Request(url, data)
response = uReq.urlopen(req)
html = response.read().decode('utf-8')
target = json.loads(html)

print('翻译结果:%s' % (target['translateResult'][0][0]['tgt']))
