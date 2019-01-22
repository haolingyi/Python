import urllib.request as uReq
import urllib.parse as uPar
import re
from bs4 import BeautifulSoup


def main():
    keyword = input('请输入关键词:')
    keyword = uPar.urlencode({'word': keyword})
    print(keyword)
    response = uReq.urlopen(
        'https://baike.baidu.com/search?%s' % keyword)
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')

    for each in soup.findAll(href=re.compile('view')):
        print(each.text)
        content = ''.join([each.text])
        url2 = ''.join(['http://baike.baidu.com', each['href']])
        response2 = uReq.urlopen(url2)
        html2 = response2.read()
        soup2 = BeautifulSoup(html, 'html.parser')
        if soup2.h2:
            content = ''.join(content, soup2.h2.text)
        content = ''.join([content, '->', url2])
        print(content)


if __name__ == '__main__':
    main()
