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
MAC_HEADERS = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}

def read_urls():
    """
    Reads through a file containing a list of all destructoid review URLS 
    Returns list of strings
    """
    list_of_review_urls = []
    with open("/Users/nick/Desktop/PSTAT 131/destructoid_urls.txt", 'r') as file:
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
        title = soup.find('h1', class_='entry-title').text.strip()
        # can use title to see portion where information about review occurs
        title_as_token = word_tokenize(title)
        counter = 0
        for index in range(len(title_as_token)):
            if title_as_token[index] == "Review" or title_as_token[index] == "review":
                break
            counter += 1
        
        title = " ".join(title_as_token[index+2:])
        review_text = soup.find("div", class_ = "entry-content")

        def is_valid(tag):
            nonlocal title
            return (tag.name == 'p' and not tag.text.startswith(title) and not tag.text.startswith("Developed by")
                    and not tag.text.startswith("Published by") and not tag.text.startswith("Released on") and not tag.text.startswith("Score")
                    and not tag.text.startswith("MSRP"))
    
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
        review_score = soup.find("div", class_="review-score").text.strip()
        return review_score
    except Exception as e:
        print(f"There was error {e} with url {url}")
        return None

def scrape_destructoid_review(url, counter):
    try:
        soup = get_html(url, WIN_HEADERS)
        review_score = get_score(soup, url)
        review_text = get_text(soup, url)
        if counter == 1:
            dataframe=pd.DataFrame({"Review Score":review_score, "Outlet" : "Gamespot" ,"Review Text":review_text}, index=[0])
            dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/destructoid_dataset.csv')
        else:
            dataframe = pd.DataFrame({"Review Score":review_score, "Outlet" : "Destructoid", "Review Text":review_text}, index=[counter-1])
            dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/destructoid_dataset.csv', header=False, mode='a')
        return True
    except Exception as e:
        print(f"Error: {e}")
        print(f"Occured for review url: {url}")
        return False
    

# def main():
#     all_reviews = read_urls()
#     progress_bar = tqdm(total=len(all_reviews), desc="Pushing Reviews to CSV")
#     for review in all_reviews:
#         soup = get_html(review)
#         if soup:
#             review_text = get_text(soup)
#             review_score = get_score(soup)
#             df = pd.DataFrame = ({"Review Score":review_score, "Review Text":review_text})
#             if os.path.exists('/Users/nick/Desktop/PSTAT 131/Project/destructoid_dataset.csv'):
#                 df.to_csv('/Users/nick/Desktop/PSTAT 131/Project/destructoid_dataset.csv', mode='a', header=False)
#             else:
#                 df.to_csv('/Users/nick/Desktop/PSTAT 131/Project/destructoid_dataset.csv')
#         progress_bar.update(1)
#     progress_bar.close()

def main():
    # https://www.destructoid.com/reviews/destructoid-review-bioshock/ "https://www.destructoid.com/reviews/review-the-jackbox-party-pack-10/"
    soup = get_html("https://www.destructoid.com/reviews/review-the-jackbox-party-pack-10/", MAC_HEADERS)
    # works with basic url 
    print(get_score(soup, url="https://www.destructoid.com/reviews/review-the-jackbox-party-pack-10/"))
    print()
    print(get_text(soup, url="https://www.destructoid.com/reviews/review-the-jackbox-party-pack-10/"))
if __name__ == "__main__":
    main()