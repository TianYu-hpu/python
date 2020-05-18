import mytest #导入mytest模块

'''
from mytest import hello
from mytest import strtest
简写成:
from mytest import hello,strtest

在讲if __name__ == '__main__'语句之前，先给大家介绍一下”程序的入口”。

Python与其他语言一样，程序都要有一个运行入口。当我们运行某个py文件时，
就能启动整个程序。那么这个py文件就是程序的入口

当然, 以后还会遇到更复杂的情况, 只有一个主模块,引入了其他多个模块。

当把mytest.py导入到main.py文件中，在mian.py中加入if __name__ == '__main__': 执行main.py, 程序正常执行。
'''

print("我是a模块")
if __name__ == '__main__':
    print("我是a模块")
    print(mytest.strtest)  # 打印mytest模块中变量strtest 

    mytest.hello()  # 运行mytest模块中函数hello()

    shaonian = mytest.Test()  # mytest模块中Test类的实例化
    print(shaonian.strClass)  # 打印实例属性
    shaonian.go()  # 调用实例方法go方法