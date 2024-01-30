# want to avoid Hardware, Movies, Tv
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests

WIN_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    


def create_urls():
    """
    create_urls creates a list of urls to use for web-scraping PCGamer reviews
    :return: list of urls being used for web-scraping
    :rtype: [str]
    """
    BASE_URL= 'https://www.gamesradar.com/reviews/archive/'    
    all_archive_links = []
    for year in range(2006, 2024):
        url = BASE_URL + str(year) + "/"
        if year == 2010:
            for month in range(6, 13):
                if month < 10:
                    myurl = url + "0" + str(month) + "/"
                else:
                    myurl = url + str(month) + "/"
                all_archive_links.append(myurl)
        else:
            for month in range(1, 13):
                if month < 10:
                    myurl = url + "0" + str(month) + "/"
                else:
                    myurl = url + str(month) + "/"
                all_archive_links.append(myurl)
    return all_archive_links

def run_through_review_pages(review_pages, headers):
    """
    run_through_review_pages sends get requests to all urls generated from create_urls in order to get specific review urls
    for scraping and data collection
    :param review_pages: List of pages containing links to individual reviews
    :type review_pages: List containing str
    :param headers: Device headers used to send get requests
    :type headers: dict
    :return: A list of all individual review URLs
    :rtype: List containing str
    """
    review_urls = []
    progress_bar = tqdm(total=len(review_pages), desc='Running through urls')
    for review_page in review_pages:
        page = requests.get(review_page, headers=headers)
        reviews = BeautifulSoup(page.content, 'html.parser')
        html = reviews.find('ul', class_='basic-list')
        for review in html.find_all("a"):
            review_urls.append(review['href'])
        progress_bar.update(1)
    progress_bar.close()
    return review_urls

def get_all_reviews():
    """
    get_all_reviews uses both create_urls and run_through_review_pages to create a file containing all review urls for scraping
    """
    all_review_pages = create_urls()
    all_reviews = run_through_review_pages(all_review_pages)
    with open("C:/Users/nicho/OneDrive/Desktop/Project/Webscrape/games_radar_urls.txt", 'w') as f:
        f.write(str(all_reviews))
