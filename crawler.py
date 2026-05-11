import logging
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import jsonlines


class HTMLDownloaderSelenium:
    def __init__(self):
        self.logger = self.__get_logger()
        self.driver = self.__get_driver()
        self.failed_links = []

    def __get_logger(self):
        logger = logging.getLogger('WowheadDownloader')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler('wowhead_downloader.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def __get_driver(self):
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Firefox(options=options)
        return driver

    def quit(self):
        self.driver.quit()

    def _find_first(self, xpaths_str):
        for xpath in [x.strip() for x in xpaths_str.split(' | ')]:
            try:
                elements = self.driver.find_elements(By.XPATH, xpath)
                if elements:
                    return elements[0].text.strip()
            except:
                continue
        return ""

    def process_url(self, url, xpath_map, prepare_page):
        results = {}
        self.logger.info(f"Processing: {url}")
        self.driver.get(url)

        if not prepare_page(self.driver):
            return None

        for key, value in xpath_map.items():
            if isinstance(value, dict):
                root_path = value['_root']
                results[key] = []

                while True:
                    time.sleep(1)
                    root_elements = self.driver.find_elements(By.XPATH, root_path)

                    for root_element in root_elements:
                        subresult = {}
                        for sub_key, xpath in value.items():
                            if sub_key.startswith("_"):
                                continue
                            text = ""
                            for single_xpath in [x.strip() for x in xpath.split(' | ')]:
                                try:
                                    elements = root_element.find_elements(By.XPATH, single_xpath)
                                    if elements:
                                        text = " ".join([e.text for e in elements]).strip()
                                        break
                                except:
                                    continue
                            subresult[sub_key] = text
                        if any(v for v in subresult.values()):
                            results[key].append(subresult)

                    if '_next_page' in value:
                        try:
                            next_btn = self.driver.find_element(By.XPATH, value['_next_page'])
                            if "disabled" in next_btn.get_attribute("class") or not next_btn.is_displayed():
                                break
                            next_btn.click()
                            time.sleep(2)
                        except:
                            break
                    else:
                        break
            else:
                results[key] = self._find_first(value)
        return results

    def get_links(self, url, css_selector):
        try:
            self.driver.get(url)
            time.sleep(2)
            dom = self.driver.page_source
            document = BeautifulSoup(dom, 'html.parser')
            links = [a.get('href') for a in document.select(css_selector) if a.get('href')]
            return [f"https://www.wowhead.com{l}" if l.startswith('/') else l for l in links]
        except:
            return []


def prepare_article_page(driver):
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h1'))
        )
        time.sleep(3)

        return True
    except Exception as e:
        print(f"Stránku článku se nepodařilo načíst: {e}")
        return False


def is_news_article_url(url):
    pattern = r'https://www\.wowhead\.com/news/[a-z0-9][a-z0-9\-]+-\d+$'
    return bool(re.match(pattern, url))


def crawl_wowhead_articles(starting_url, xpath_map, output_file="wowhead_articles.jsonl", max_pages=20):
    downloader = HTMLDownloaderSelenium()
    visited = set()
    all_links = []

    for page in range(1, max_pages + 1):
        page_url = starting_url if page == 1 else f"{starting_url}?page={page}"
        downloader.logger.info(f"Načítám seznam článků – stránka {page}: {page_url}")

        raw_links = downloader.get_links(page_url, 'a[href*="/news/"]')
        new_links = [url for url in raw_links if is_news_article_url(url) and url not in all_links]

        if not new_links:
            downloader.logger.info(f"Stránka {page} neobsahuje žádné nové články – končím stránkování.")
            break

        all_links.extend(new_links)
        downloader.logger.info(f"Stránka {page}: nalezeno {len(new_links)} nových článků (celkem {len(all_links)}).")

    queue = list(dict.fromkeys(all_links))
    downloader.logger.info(f"Celkem nalezeno {len(queue)} unikátních článků ke stažení.")

    results = []
    for url in queue:
        if url in visited:
            continue

        visited.add(url)
        result = downloader.process_url(url, xpath_map, prepare_article_page)

        if result:
            result['url'] = url
            results.append(result)
            with jsonlines.open(output_file, mode='a') as writer:
                writer.write(result)
            downloader.logger.info(f"Uložen článek: {url}")

    downloader.quit()
    return results


if __name__ == "__main__":
    wowhead_article_xpath_map = {
        # Titulek článku
        'title': '//h1[@class="heading-size-1"] | //h1[contains(@class,"heading")] | //h1',

        # Autor článku
        'author': '//div[contains(@class,"posted-by")]//a | //span[contains(@class,"author")] | //a[contains(@class,"author")]',

        # Datum publikace
        'date': '//div[contains(@class,"posted-by")]//span[@class="date"] | //time[@class="date"] | //span[contains(@class,"date")]',

        # Celý textový obsah článku
        'content': '//div[contains(@class,"news-post-content")]',
    }
    start_url = 'https://www.wowhead.com/news'
    crawl_wowhead_articles(start_url, wowhead_article_xpath_map, output_file="wowhead_articles.jsonl", max_pages=50)

