import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

"""
GamespotGetURLs.py creates a file of all gamespot review's associated urls for web-scraping
"""
WIN_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    


def create_urls():
    """
    create_urls creates a list of urls to use for web-scraping Gamespot reviews
    :return: list of urls being used for web-scraping
    :rtype: [str]
    """
    # creates all 735 pages necessary to gather all gamespot reviews recorded
    BASE_URL = 'https://www.gamespot.com/games/reviews/?page='
    counter = 1
    URLS = []
    while counter < 739:
        url = BASE_URL + str(counter)
        print(url)
        counter += 1
        URLS.append(url)
    return URLS

def run_through_urls(urls, headers):
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

    # Headers for accessing the gamespot web-page
    review_urls = []
    progress_bar = tqdm(total=len(urls), desc='Running through urls')

    for url in urls:
        # For every url we send a get request to get the page
        page = requests.get(url, headers=headers)

        # Every page has around 21 reviews that can be found within the 'editorial river' segment of html
        try:
            reviews = BeautifulSoup(page.content, 'html.parser').find("section", class_="editorial river").find_all("a", class_="card-item__link")

            # For every review in the 'editorial river' segment we find its 'href' tag and append that to a list
            for review in reviews:
                review_urls.append(review['href'])    
        except:
            print(f'problem with the following {url}')
        progress_bar.update(1)
    progress_bar.close()
    return review_urls


def get_all_reviews():
    """
    get_all_reviews uses both create_urls and run_through_review_pages to create a file containing all review urls for scraping
    """
    urls = create_urls()
    all_urls = run_through_urls(urls)
    with open('C:/Users/nicho/OneDrive/Desktop/gamespot_urls.txt','w') as f:
        f.write(str(all_urls))

    