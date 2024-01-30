from bs4 import BeautifulSoup
import pandas as pd
import requests 
import os.path
from tqdm import tqdm
from nltk.tokenize import word_tokenize



"""
"""

WIN_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    
MAC_HEADERS = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}

def read_urls():
    """
    Reads through a file containing a list of all destructoid review URLS 
    Returns list of strings
    """
    list_of_review_urls = []
    with open("C:\\Users\\nicho\\OneDrive\\Desktop\\Project\\Webscrape\\games_radar_urls.txt", 'r') as file:
        for line in file:
            list_of_review_urls.append(line)
    return list_of_review_urls

def get_html(url, device_headers):
    """
    Takes in a URL which requests uses to get the HTML of the page as a BS4 object
    Returns BS4 soup object
    """
    page_html = requests.get(url, headers=device_headers)
    return BeautifulSoup(page_html.content, 'html.parser')

def get_text(soup, url):
    """
    Parses through BS4 object to get review text
    Returns text
    """
    try:

        def check_validity(tag):
            return tag.name == "p" and not tag.find_parents('div', class_="fancy_box_body") and not tag.find("em")
        # print(review_body)
        final_text = ""
        review_text = soup.find("div", class_="text-copy bodyCopy auto").find_all(check_validity)
        for line in review_text:
            final_text += (line.text.strip() + "\n")
        return final_text
    
    except Exception as e:
        print(f"There was error {e} with url {url}")
        return None     

def get_title(soup, url):
    """
    get_title gets the title from the reviewed item to avoid duplicates
    """
    try:
        def check_validity(tag):
            return tag.name == "h1"
        print(soup.find(check_validity))
        return soup.find(check_validity).text.strip()
    except Exception as e:
        print(f"Error {e} occured for url {url}: could not get title")
        return None

def get_score(soup, url):

    """
    Parses through BS4 object to find review score
    Returns score
    """
    try:
        review_chunk = soup.find("span", class_="chunk rating")
        full_stars = review_chunk.find_all("span", class_="icon icon-star")
        half_stars = review_chunk.find_all("span", class_="icon icon-star half")
        if half_stars is not []:
            score = len(full_stars) + len(half_stars)/2
        else:
            score = len(full_stars)
        return score * 2
    
    except Exception as e:
        print(f"There was error {e} with url {url}")
        return None

def scrape_games_radar_review(url, counter):
    check = ['Tech', 'Asus ROG Phone 3', 'Hardware', 'Sports', 'IPone', 'Google Stadia', 'Sparta', 'Android', 'Toys 1 Wireless', 'Toys', 'BenQ Zowie XL2546K', 'Dell S2721DGF', 'Family', 'TV', 'Tabletop Gaming', 'Razer Nommo V2 Pro', 'SteelSeries Sensei Ten', 'Comics', 'Movies']
    try:
        soup = get_html(url, WIN_HEADERS)
        signifier = soup.find('nav', class_='breadcrumb').text.strip().split('\n')[0]
        if signifier not in check:
            review_score = get_score(soup, url)
            review_text = get_text(soup, url)
            review_title = get_title(soup, url)
            # print(review_score, review_text,[review_title])
            if counter == 1:
                dataframe=pd.DataFrame({"Review Score":review_score, "Outlet" : "GamesRadar", "Title":review_title, "Signifier" : signifier, "Review Text":review_text}, index=[0])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/gamesradar_dataset.csv')
            else:
                dataframe = pd.DataFrame({"Review Score":review_score, "Outlet" : "GamesRadar", "Signifier" : signifier, "Title":review_title, "Review Text":review_text}, index=[counter-1])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/gamesradar_dataset.csv', header=False, mode='a')
            return True
    except Exception as e:
        print(f"Error: {e}")
        print(f"Occured for review url: {url}")
        return False
    
def main():
    scrape_games_radar_review('https://www.gamesradar.com/portal-3/', 1)
    scrape_games_radar_review('https://www.gamesradar.com/portal-review/', 1)


# if __name__ == "__main__":
#     main()