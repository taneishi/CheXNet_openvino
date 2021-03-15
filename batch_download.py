# Download the 12 tgz files in batches
import urllib.request
import os

# URLs for the tar.gz files
links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]

print('%d files will be downloaded.' % len(links))
for index, link in enumerate(links, 1):
    filename = 'images_%02d.tar.gz' % index
    print(filename)
    if not os.path.exists(filename):
        print('downloading %s...' % filename)
        urllib.request.urlretrieve(link, filename) # download the tgz file

print('Download complete. Please check the checksums')

sha256sum = [
        'fd8e3542db6351ae9377779033f5d5c5f32fe50eb0830b519fbf1a7e791354b1',
        'c849cfa5504b8cffb301952bf93d53d3d39d7d931bc88bb70427b4a67de0740a',
        '6a90a979850545e30ed3e9ba96de2f063db13aa2a3c22363820d3f70d8c882c8',
        '1ca953ec37fe9c132ec49743f9651b707884c6c76e6e3a33853940c13372a385',
        'a058c365478454dc7395ee2897c2b63a3bf108739b857046838859227e42e743',
        '9b00987d39e4c9e4ab18b7426db82197c043dacec5a827141446c3900d39d5dd',
        '7868b43798cf1bc9ae98d142644864b056e64edda3be12f7504f16485ac382ff',
        '823088934d0ea57b4a7384446560bcdb825e712932b69e8541a84acee10af447',
        'bdd5bfe76323b630f5f83a013a0f2f100d1b03fdaea25240844e8763c1e40a51',
        '8f18699d3baad03cdf8946220a95e8d2bab555dc8e4be99e13bc0366a72aba1c',
        '7e7d190f71b5b792c495acb2eebbbf0563a8c528adb23e9335ec3b79cb5b486f',
        '7316ce5f4d5e0154730e592cc2b45b48a2bee8457f6339d36c2ca55ed6e60b26',
        ]
print(sha256sum)
