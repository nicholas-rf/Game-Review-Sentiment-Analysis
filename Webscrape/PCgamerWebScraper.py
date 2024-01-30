# how functions get imported from another .py file is actually pretty self-explanatory
# this project should be a destructoid web-scraper
from bs4 import BeautifulSoup
import pandas as pd
import requests
import os.path
from tqdm import tqdm
from nltk.tokenize import word_tokenize

"""
"""

WIN_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    


def read_urls():
    """
    Reads through a file containing a list of all destructoid review URLS 
    Returns list of strings
    """
    list_of_review_urls = []
    with open("/Users/nick/Desktop/PSTAT 131/pc_gamer_urls.txt", 'r') as file:
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

def check_validity(soup):
    # print(soup.find("div", class_="box contrast less-space pro-con"), soup.find_all(check_for_specifications), soup.find_all(check_for_benchmark))
    # print(soup.find("div", class_="box contrast less-space pro-con") is None and len(soup.find_all(check_for_specifications)) == 0 and len(soup.find_all(check_for_benchmark)) ==0)
    return soup.find("div", class_="box contrast less-space pro-con") is None and len(soup.find_all(check_for_specifications)) == 0 and len(soup.find_all(check_for_benchmark)) ==0

def check_for_specifications(tag):
    return (tag.name == 'p' and tag.find('strong', string = ' Specifications ')) or (tag.name == 'p' and tag.find('strong', string = ' Specifications: '))

def check_for_benchmark(tag):
    return (tag.name == 'p' and tag.find('strong', string=' Benchmarks ')) or (tag.name == 'p' and tag.find('strong', string=' Benchmarks: '))

def get_text(soup, url):
    """
    Parses through BS4 object to get review text
    Returns text
    """
    try:
        review_text = soup.find('div', id='article-body')

        def is_valid(tag):
            return (tag.name == 'p' and not tag.name == 'fancy_box_title' and not tag.find_parents('figure')
                    and not tag.text.startswith("What is it?") and not tag.text.startswith("Release:") and not tag.text.startswith("Expect to pay:")
                    and not tag.text.startswith("Developer:") and not tag.text.startswith("Publisher:") and not tag.text.startswith("Multiplayer:")
                    and not tag.text.startswith("Link:"))
        
        review_text = review_text.find_all(is_valid) 
        final_text = ""
        for line in review_text:
            final_text += (line.text.strip() + "\n")
        return final_text
    except Exception as e:
        print(f"There was error {e} with url {url}")
        return None

def get_score(soup, url):
    """
    Parses through BS4 object to find review score
    Returns score
    """
    try:
        review_score = soup.find("div", class_="review-score-wrapper review-score-pcg-generic").text.strip()
        return float(review_score) / 10 
    except Exception as e:
        print(f"There was error {e} with url {url}")
        return None
    
def scrape_pc_gamer_review(url, counter):
    """
    Gets reviews from PC Gamer
    """
    try:
        soup = get_html(url, WIN_HEADERS)

        if check_validity(soup):
            review_score = get_score(soup, url)
            review_text = get_text(soup, url)
            # print(review_score, review_text)
            if counter == 1:
                print("pushing_to_csv")
                dataframe=pd.DataFrame({"Review Score":review_score, "Outlet" : "PCG" ,"Review Text":review_text}, index=[0])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/pc_gamer_dataset.csv')
            else:
                dataframe = pd.DataFrame({"Review Score":review_score, "Outlet" : "PCG", "Review Text":review_text}, index=[counter-1])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/pc_gamer_dataset.csv', header=False, mode='a')
            return True
        else:
            # print(f"review at {url} is causing issue")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Occured for review url: {url}")
        return False
 
# scrape_pc_gamer_review("https://www.pcgamer.com/sbk-x-review/", 1)
# https://www.pcgamer.com/amd-radeon-hd-7990-6gb-review/
# https://www.pcgamer.com/corsair-vengeance-6182-gaming-desktop-review/
# https://www.pcgamer.com/sbk-x-review/
# Cleaning up for destructoid reviews includes getting rid of [reviewed] sentence, might be better to just stick with developer: tag
# and getting rid of elements like [this review is based on a retail build of the game provided by the publisher]