import scrapy


class MovieSpider(scrapy.Spider):
    name = 'imdb'

    start_urls = ['https://www.imdb.com/chart/tvmeter/?ref_=nv_tvv_mptv']

    def parse(self, response):
        lista = []
        for item in response.xpath('//td[@class="titleColumn"]'):
            movie_page = item.xpath('.//a/@href').extract_first()
            lista.append(movie_page)
        for i in lista:
            next_page_link = response.urljoin(i)
            yield scrapy.Request(url=next_page_link, callback=self.parse2)


    def parse2(self, response):
        yield {
            'title': (response.xpath('//div[@class="title_wrapper"]/h1/text()').extract_first()).strip(),
            'rating': response.xpath('//div[@class="ratingValue"]/strong/span/text()').extract_first(),
            'creator': response.xpath('//div[@class="credit_summary_item"]/a/text()').extract_first(),
            'seasons': response.xpath('//div[@class="seasons-and-year-nav"]/div/a/text()').extract_first(),
            'genres': response.xpath('//div[@class="title_wrapper"]/div/a/text()').getall()
        }

