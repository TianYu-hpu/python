# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
#导入scrapy
import scrapy

'''
我们导入了scrapy。因为我们等会所创建的类将直接继承scrapy中的scrapy.Item类。这样，有很多好用属性和方法，就能够直接使用。比如，引擎能将item类的对象发给Item Pipeline（数据管道）处理。这些可能你暂时听不懂，但是学完这课你就明白这整个流程了。

我们定义了一个BookDoubanItem类。它继承自scrapy.Item类。

我们定义了书名、出版信息和评分三种数据。scrapy.Field()这行代码实现的是，让数据能以类似字典的形式记录。下面的代码展示了一下如何使用 BookDoubanItem：
'''
class BookDoubanItem(scrapy.Item):
#定义一个类BookDoubanItem，它继承自scrapy.Item
    title = scrapy.Field()
    #定义书名的数据属性
    publish = scrapy.Field()
    #定义出版信息的数据属性
    score = scrapy.Field()
    #定义评分的数据属性