#导入所需模块
import scrapy
import bs4
from ..items import Class14Item
'''
这里我需要向你解释一下第31行代码的含义：scrapy.Request是构造requests对象的类。real_url是我们往requests对象里传入的每家公司招聘信息网址的参数。
callback的中文意思是回调。self.parse_GetJobInfo是我们新定义的parse_GetJobInfo方法。往requests对象里传入callback=self.parse_GetJobInfo这个参数后，引擎就能知道response要前往的下一站，是用parse_GetJobInfo()方法来解析传入的新参数。

在这里yield的作用就是把这个构造好的requests对象传递给引擎
第33行代码，是我们定义的新的parse_GetJobInfo方法。这个方法是用来解析和提取公司招聘信息的数据。
对照一下我们上文对如何获取招聘信息的数据定位表格，你应该能比较好地理解33-54行代码

第33-54行代码：提取出公司名称、职位名称、工作地点和招聘要求这些数据，并把这些数据放进我们定义好的Class14Item类里。
最后，用yield语句把item传递给引擎。至此，爬虫部分的核心代码就写完啦

该如何保存我们爬取到的数据呢？在第6关，我们学过用csv模块把数据存储csv文件，用openpyxl模块把数据存储Excel文件
Scrapy可以支持把数据存储成csv文件或者Excel文件，当然实现方式是不一样的。
我们先来看如何讲数据保存为csv文件。存储成csv文件的方法比较简单，只需在settings.py文件里，添加如下的代码即可。

FEED_URI='%(name)s.csv'
FEED_FORMAT='CSV'
FEED_EXPORT_ENCODING='ansi'
FEED_URI是导出文件的路径。'%(name)s.csv'，就是把CSV文件放到与settings.py文件同级文件夹内。
FEED_FORMAT 是导出数据格式，写CSV就能得到CSV格式。
FEED_EXPORT_ENCODING 是导出文件编码，ansi是一种在windows上的编码格式，你也可以把它变成utf-8用在mac电脑上。

再来看如何把数据存储成Excel文件。这个存储的方法要稍微复杂一些，我们需要先在setting.py里设置启用ITEM_PIPELINES，设置方法如下：

#需要修改`ITEM_PIPELINES`的设置代码：

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#      'class14.pipelines.Class14Pipeline': 300,
# }

'''
class Spider_GetJobInfo(scrapy.Spider):
#定义一个爬虫类Spider_GetJobInfo
    name = 'GetJobsInfo'
    #定义爬虫的名字为GetJobs
    allowed_domains = ['www.jobui.com']
    #定义允许爬虫爬取网址的域名——职友集网站的域名
    start_urls = ['https://www.jobui.com/rank/company/']
    #定义起始网址——职友集企业排行榜的网址

    def parse(self, response):
    #parse是默认处理response的方法
        bs = bs4.BeautifulSoup(response.text, 'html.parser')
        #用BeautifulSoup解析response（企业排行榜的网页源代码）
        ul_list = bs.find_all('ul',class_="textList flsty cfix")
        #用find_all提取<ul class_="textList flsty cfix">标签
        for ul in ul_list:
        #遍历ul_list
            a_list = ul.find_all('a')
            #用find_all提取出<ul class_="textList flsty cfix">元素里的所有<a>元素
            for a in a_list:
            #再遍历a_list
                company_id = a['href']
                #提取出所有<a>元素的href属性的值，也就是公司id标识
                url = 'https://www.jobui.com{id}jobs'.format(id=company_id)
                #构造出包含公司名称和招聘信息的网址链接的list
                yield scrapy.Request(url, callback=self.parse_GetJobInfo)
    # 用yield语句把构造好的request对象传递给引擎。用scrapy.Request构造request对象。callback参数设置调用parse_GetJobInfo方法。

    def parse_GetJobInfo(self, response):
        # 定义新的处理response的方法parse_GetJobInfo（方法的名字可以自己起）
        bs = bs4.BeautifulSoup(response.text, 'html.parser')
        # 用BeautifulSoup解析response(公司招聘信息的网页源代码)
        company = bs.find(class_="company-banner-name").text
        # 用find方法提取出公司名称
        datas = bs.find_all('div', class_="job-simple-content")
        # 用find_all提取<div class_="job-simple-content">标签，里面含有招聘信息的数据
        for data in datas:
            # 遍历datas
            item = Class14Item()
            # 实例化Class14Item这个类
            item['company'] = company
            # 把公司名称放回Class14Item类的company属性里
            item['position'] = data.find_all('div', class_="job-segmetation")[0].find('h3').text
            # 提取出职位名称，并把这个数据放回Class14Item类的position属性里
            item['address'] = data.find_all('div', class_="job-segmetation")[1].find_all('span')[0].text
            # 提取出工作地点，并把这个数据放回Class14Item类的address属性里
            item['detail'] = data.find_all('div', class_="job-segmetation")[1].find_all('span')[1].text
            # 提取出招聘要求，并把这个数据放回Class14Item类的detail属性里
            yield item
            # 用yield语句把item传递给引擎








