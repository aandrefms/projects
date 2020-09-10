import scrapy
API = 'KEY'
from urllib.parse import urlencode


def get_url(url):
    payload = {'api_key': API, 'url': url}
    proxy_url = 'http://api.scraperapi.com/?' + urlencode(payload)
    return proxy_url

queries = ['https://www.amazon.com.br/s?rh=n%3A6740748011%2Cn%3A%217841278011%2Cn%3A7872854011&page=2&qid=1599402849&ref=lp_7872854011_pg_2']


class BooksSpider(scrapy.Spider):

    name = 'books'

    def start_requests(self):
        for query in queries:
            url = query
            yield scrapy.Request(url=get_url(url), callback=self.parse_keyword_response)

    def parse_keyword_response(self, response):
        books = response.xpath('//div[@class="sg-col-20-of-24 s-result-item s-asin sg-col-0-of-12 sg-col-28-of-32 sg-col-16-of-20 sg-col sg-col-32-of-36 sg-col-12-of-16 sg-col-24-of-28"]')
        for item in books:
            yield {
                'asin' : item.xpath('@data-asin').extract_first(),
                'title' : item.xpath('.//span[@class="a-size-medium a-color-base a-text-normal"]/text()').extract_first(),
                'price_normal': item.xpath('.//span[@class="a-price"]/span/text()').extract_first(),
                'author': item.xpath('.//span[@class="a-size-base"]/text()').getall()[1:(len(item.xpath('.//span[@class="a-size-base"]/text()').getall()))-1],
                'total_ratings': item.xpath('.//span[@class="a-size-base"]/text()').getall()[-1]
            }


        next_page = response.xpath('//li[@class="a-last"]/a/@href').extract_first()
        print(next_page)
        next_page = f'https://www.amazon.com.br{next_page}'
        if next_page:
            next_page_link = response.urljoin(next_page)
            yield scrapy.Request(url=get_url(next_page_link), callback=self.parse_keyword_response)



