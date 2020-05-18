import scrapy
import bs4
 
'''
导入BeautifulSoup用于解析和提取数据;导入scrapy是待会我们要用创建类的方式写这个爬虫，
我们所创建的类将直接继承scrapy中的scrapy.Spider类。这样，有许多好用属性和方法，就能够直接使用。

第17行代码：定义一个爬虫类DoubanSpider。就像我刚刚讲过的那样，DoubanSpider类继承自scrapy.Spider类。
第18行代码：name是定义爬虫的名字，这个名字是爬虫的唯一标识。name = 'book_douban'意思是定义爬虫的名字为book_douban。等会我们启动爬虫的时候，要用到这个名字。
第19行代码：allowed_domains是定义允许爬虫爬取的网址域名（不需要加https://）。如果网址的域名不在这个列表里，就会被过滤掉。
为什么会有这个设置呢？当你在爬取大量数据时，经常是从一个URL开始爬取，然后关联爬取更多的网页。假设我们在今天的爬虫目标是爬豆瓣top250每本书籍的书评信息，我们会先爬取书单，再找到每本书的URL，再进入每本书的详情页面去抓取评论。
allowed_domains就限制了我们这种关联爬取的URL，一定在book.douban.com这个域名之下，不会跳转到某个奇怪的广告页面。
第20行代码：start_urls是定义起始网址，就是爬虫从哪个网址开始抓取。在此，allowed_domains的设定对start_urls里的网址不会有影响。
第21行代码：parse是Scrapy里默认处理response的一个方法

'''

class DoubanSpider(scrapy.Spider):
    name = 'book_douban'
    allowed_domains = ['book.douban.com']
    #定义爬虫爬取网址的域名。
    start_urls = []
    #定义起始网址。
    for x in range(3):
        url = 'https://book.douban.com/top250?start=' + str(x * 25)
        start_urls.append(url)
        #把豆瓣Top250图书的前3页网址添加进start_urls。

    def parse(self, response):
    #parse是默认处理response的方法。
        bs = bs4.BeautifulSoup(response.text,'html.parser')
        #用BeautifulSoup解析response。
        datas = bs.find_all('tr',class_="item")
        #用find_all提取<tr class="item">元素，这个元素里含有书籍信息。
        for data in  datas:
        #遍历data。
            title = data.find_all('a')[1]['title']
            #提取出书名。
            publish = data.find('p',class_='pl').text
            #提取出出版信息。
            score = data.find('span',class_='rating_nums').text
            #提取出评分。
            print([title,publish,score])
            #打印上述信息。
            # 遍历data
            item = BookDoubanItem()
            # 实例化DoubanItem这个类
            item['title'] = data.find_all('a')[1]['title']
            # 提取出书名，并把这个数据放回DoubanItem类的title属性里
            item['publish'] = data.find('p', class_='pl').text
            # 提取出出版信息，并把这个数据放回DoubanItem类的publish里
            item['score'] = data.find('span', class_='rating_nums').text
            # 提取出评分，并把这个数据放回DoubanItem类的score属性里。
            print(item['title'])
            # 打印书名
            '''
             在Scrapy框架里，每一次当数据完成记录，它会离开spiders，来到Scrapy Engine（引擎），
             引擎将它送入Item Pipeline（数据管道）处理。这里，要用到yield语句。
             yield语句你可能还不太了解，这里你可以简单理解为：它有点类似return，不过它和return不同的点在于，它不会结束函数，且能多次返回信息
             
             用可视化的方式来呈现程序运行的过程，就如同上图所示：爬虫（Spiders）会把豆瓣的10个网址封装成requests对象，
             引擎会从爬虫（Spiders）里提取出requests对象，再交给调度器（Scheduler），让调度器把这些requests对象排序处理。
             
             然后引擎再把经过调度器处理的requests对象发给下载器（Downloader），下载器会立马按照引擎的命令爬取，并把response返回给引擎。
             
             紧接着引擎就会把response发回给爬虫（Spiders），这时爬虫会启动默认的处理response的parse方法，
             解析和提取出书籍信息的数据，使用item做记录，返回给引擎。引擎将它送入Item Pipeline（数据管道）处理。
             '''
            yield item
            # yield item是把获得的item传递给引擎