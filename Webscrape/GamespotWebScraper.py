import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import pickle

# URL's [https://www.gamespot.com/reviews/cocoon-review-a-bugs-strife/1900-6418123/] [https://www.gamespot.com/reviews/starfield-review-to-infinity-but-not-beyond/1900-6418110/]
# Error URL [https://www.gamespot.com/reviews/bad-mojo-review/1900-2538115/]

"""
GamespotWebScraper.py scrapes all gamespot review URLs for their content
"""

def check_validity(soup, url):
    """
    check_validity makes sure that the review at url can be scraped properly
    :param soup: A BS4 object containing a reviews html
    :type soup: BS4 object
    :param url: A url pointing to the review of the soup object
    :type url: str
    :return: Either True or False depending on if the review is scrapable
    :rtype: boolean
    """
    # Attempts to find the body segment of the html
    body_check = soup.find('body')
    try:
        # If body check doesn't have a body-error then the review is scrapable
        if body_check['class'] == ["body-error"]:
            return False
        else:
            return True
    except KeyError:
        print("Issue occured at the following URL: ", url, "\n")
        print("Check to make sure the HTML is compatible with the webscraper")
        return False
    except:
        print("Issue occured at the following URL: ", url, "\n")
        print("Might not be HTML issue")
        return False

def get_html(url='https://www.gamespot.com/reviews/cocoon-review-a-bugs-strife/1900-6418123/'):
    """
    get_html takes a url and returns a BS4 object with the urls html
    :param url: A review url
    :type url: str
    :return: A BS4 object with urls html
    :rtype: BS4 object
    """
    headers = {"User-Agent": 
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    
    page = requests.get(url, headers=headers)
    return BeautifulSoup(page.content,'html.parser')

def get_score(soup, url):
    """
    get_score takes a urls beautiful soup object and returns the urls review score
    :param soup: A BS4 object containing a reviews html
    :type soup: BS4 object
    :param url: A url pointing to the review of the soup object
    :type url: str
    :return: A score from the review
    :rtype: str
    """
    # Finds the review score within the 'review-ring-score__score text-bold' portion of the html
    score_in_soup = soup.find(class_="review-ring-score__score text-bold")
    try:
        # Attempts to return the score
        score = score_in_soup.text.strip()
        return str(score)
    except:
        # If an issue occurs the reviews url is returned so that further inspection can occur
        print("There was an issue with score being found at following URL: ", url)
        score = "issue with url: " + url
        return score

def get_blurbs(soup, url):
    """
    get_blurbs finds blurbs within the html that arent part of the regular reviews <p> segments, and returns them
    :param soup: A BS4 object containing a reviews html
    :type soup: BS4 object
    :param url: A url pointing to the review of the soup object
    :type url: str
    :return: A start blurb found in the 'news' deck portion of the article, and a final blurb found near the review score
    :rtype: str, str
    """
    try:
        # Uses .find to get both blurb objects
        start_blurb = soup.find("p", class_="news-deck").text.strip()
        final_blurb = ""
        breakdown_blurb = soup.find("div", class_="review-breakdown__lists").find_all("li")

        # Since the breakdown blurb is a list of items, iteration provides access to all blurbs
        for blurb in breakdown_blurb:
            final_blurb += (blurb.text.strip() + "\n")
        return start_blurb, final_blurb
    except Exception as e:
        print("There was an issue with blurb being found at following URL: ", url)
        print(f"Error was {e}")
        start_blurb = "issue with url: " + url
        final_blurb = " "
        return start_blurb, final_blurb


def get_text(soup, url):    
    """
    get_text finds all of the text from the review found in <p> segments of the html, and returns it as a single string
    :param soup: A BS4 object containing a reviews html
    :type soup: BS4 object
    :param url: A url pointing to the review of the soup object
    :type url: str
    :return: The text of the review
    :rtype: str
    """
    try:
        # Finds the review text within the 'js-content-entity-body' portion of the html
        review_text = soup.find("div", class_="js-content-entity-body")
        review_text = review_text.find_all("p")
        final_text = ""
        # Iterates through all <p> segments within 'js-content-entity-body' to get the review text
        for line in review_text:
            final_text += (line.text.strip() + "\n")
        blurb1, blurb2 = get_blurbs(soup, url)
        final_text += blurb1 + blurb2
        return final_text
    
    except Exception as e:
        print("There was an issue with text being found at following URL: ", url)
        print(f'Error was {e}')
        review_text = ''
        return review_text

def scrape_gamespot_review(url, counter):
    """
    scrape_gamespot_review takes in a url and a counter and scrapes the url for its text and score,
    """

    # Soup is gathered from the url
    try:
        soup = get_html(url)
        if check_validity(soup, url):

            # Review content is obtained from the soup
            review_score = get_score(soup, url)
            review_text = get_text(soup, url)

            # Counter is used to determine if a new file needs to be made
            if counter == 1:
                dataframe=pd.DataFrame({"Review Score":review_score, "Outlet" : "Gamespot" ,"Review Text":review_text}, index=[0])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/gamespot_dataset.csv')

            # When counter is not 1, we use mode='a' to append a new row to the overall csv
            else:
                dataframe=pd.DataFrame({"Review Score":review_score, "Outlet" : "Gamespot", "Review Text":review_text}, index=[counter-1])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/gamespot_dataset.csv', header=False, mode='a')
            return True
        else:
            print(f"Occured for review url: {url}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        print(f"Occured for review url: {url}")
        return False


def main():

    # Filepath leading to file of all urls, file gets written to avoid unnecesary runtime    
    filepath = 'C:/Users/nicho/OneDrive/Desktop/list_of_all_review_urls.pk1'
    with open(filepath, 'rb') as file2:
        all_game_urls = pickle.load(file2)

    # TQDM progress bar gets initialized to track progress and a counter is initialized to keep track of index within pandas dataframe as well as to initialize csv file
    counter = 1
    progress_bar = tqdm(total=len(all_game_urls), desc="Getting Reviews")

    # Iteration through all reviews begins
    for url in all_game_urls:
        
        # Soup is gathered from the url
        soup = get_html(url)
        if check_validity(soup, url):

            # Review content is obtained from the soup
            review_score = get_score(soup, url)
            review_text = get_text(soup, url)

            # Counter is used to determine if a new file needs to be made
            if counter == 1:
                dataframe=pd.DataFrame({"Review Score":review_score, "Review Text":review_text}, index=[0])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/dataset.csv')

            # When counter is not 1, we use mode='a' to append a new row to the overall csv
            else:
                dataframe=pd.DataFrame({"Review Score":review_score, "Review Text":review_text}, index=[counter-1])
                dataframe.to_csv('C:/Users/nicho/OneDrive/Desktop/dataset.csv', header=False, mode='a')
            counter += 1    
        else:
            pass
        progress_bar.update(1)
    progress_bar.close()

# if __name__ == "__main__":
#     main()
    