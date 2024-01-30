import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def create_urls():
    """
    create_urls creates a list of urls to use for web-scraping destructoid reviews
    :return: list of urls being used for web-scraping
    :rtype: [str]
    """
    BASE_URL= 'https://www.destructoid.com/reviews/page/'
    page_no = 1
    review_pages = []
    while page_no < 342:
        if page_no == 1:
            url = 'https://www.destructoid.com/reviews/'
        else:
            url = BASE_URL + str(page_no) + "/"
        review_pages.append(url)
        page_no += 1
    return review_pages

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
        for review in reviews.find_all("a", class_="post-title"):
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
    with open("C:/Users/nicho/OneDrive/Desktop/destructoid_urls.txt", 'w') as f:
        f.write(str(all_reviews))
