import urllib.request as uReq

req = uReq.Request('http://placekitten.com/200/300')
response = uReq.urlopen(req)
catImg = response.read()
with open('catPic.jpeg', 'wb') as f:
    f.write(catImg)
