## 运行
想要运行Scrapy有两种方法，
### 一种
在本地电脑的终端跳转到scrapy项目的文件夹（跳转方法：cd+文件夹的路径名），然后输入命令行：scrapy crawl book_douban（book_douban就是我们爬虫的名字）。
### 第二种
另一种运行方式需要我们在最外层的大文件夹里新建一个main.py文件（与scrapy.cfg同级）。
我们只需要在这个main.py文件里，输入以下代码，点击运行，Scrapy的程序就会启动。
'''
from scrapy import cmdline

#导入cmdline模块,可以实现控制终端命令行

cmdline.execute(['scrapy','crawl','book_douban'])

#用execute（）方法，输入运行scrapy的命令
'''
第1行代码：在Scrapy中有一个可以控制终端命令的模块cmdline。导入了这个模块，我们就能操控终端。
第3行代码：在cmdline模块中，有一个execute方法能执行终端的命令行，不过这个方法需要传入参数。我们输入运行Scrapy的代码scrapy crawl book_douban，就需要写成['scrapy','crawl','book_douban']这样。

