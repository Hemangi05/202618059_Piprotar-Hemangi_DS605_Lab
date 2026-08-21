import scrapy

class BooksSpider(scrapy.Spider):
    name = "books"

    allowed_domains = ["books.toscrape.com"]

    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    count = 0
    max_books = 100

    def parse(self, response):

        books = response.css(
            "article.product_pod h3 a::attr(href)"
        ).getall()

        for book in books:

            if self.count >= self.max_books:
                return

            self.count += 1

            yield response.follow(
                book,
                callback=self.parse_book
            )

        next_page = response.css(
            "li.next a::attr(href)"
        ).get()

        if next_page and self.count < self.max_books:
            yield response.follow(
                next_page,
                callback=self.parse)

    def parse_book(self, response):

        title = response.css("div.product_main h1::text").get()

        price = response.css("p.price_color::text").get()

        rating = response.css(
            "p.star-rating::attr(class)"
        ).get().split()[-1]

        availability = response.xpath(
            'normalize-space(//p[contains(@class,"availability")])'
        ).get()

        description = response.xpath(
            '//div[@id="product_description"]/following-sibling::p/text()'
        ).get()

        if description is None:
            description = "No description available"

        category = response.xpath(
            '//ul[@class="breadcrumb"]/li[3]/a/text()'
        ).get()

        details = {}

        rows = response.xpath("//table//tr")

        for row in rows:
            key = row.xpath("./th/text()").get()
            value = row.xpath("./td/text()").get()
            details[key] = value

        yield {
            "Title": title,
            "Category": category,
            "Price": price,
            "Rating": rating,
            "Availability": availability,
            "Description": description,
            "UPC": details.get("UPC"),
            "Reviews": details.get("Number of reviews"),
            "URL": response.url
        }