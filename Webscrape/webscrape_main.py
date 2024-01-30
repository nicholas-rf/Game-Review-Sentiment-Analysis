import GamespotWebScraper
import DestructoidWebScraper
import PCgamerWebScraper
from tqdm import tqdm
import pcgamer_get_urls
import DestructoidGetURLs
import GamespotGetURLs
import pandas as pd
import gamesradarscraper
import get_games_radar_urls

WIN_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"}    
MAC_HEADERS = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}

pc_gamer_reviews = 'C:/Users/nicho/OneDrive/Desktop/Project/Webscrape/pc_gamer_urls.txt'
gamespot_reviews = 'C:/Users/nicho/OneDrive/Desktop/Project/Webscrape/gamespot_urls.txt'
destructoid_reviews = 'C:/Users/nicho/OneDrive/Desktop/Project/Webscrape/destructoid_urls.txt'
def create_datasets():
    # want to open both files containing review urls
    # maintain a set to get non-duplicate 
    """
    This writes down the dataset from the gamespot and destructoid web-scrapers, it uses sets to 
    avoid duplicate visits to urls that could be an issue as a result of web-scraping
    """
    with open(gamespot_reviews,'r') as gamespot_urls_f, open(destructoid_reviews, 'r') as destructoid_urls_f, open(pc_gamer_reviews, 'r') as pc_gamer_urls_f:

        print("Starting Gamespot Reviews")
        base = 'https://www.gamespot.com'
        visited_gamespot = set()
        gamespot_urls = gamespot_urls_f.read()
        gamespot_urls = eval(gamespot_urls)

        # 13823 corresponds to 13795 index, 
        counter=1
        progress_bar = tqdm(total=len(gamespot_urls), desc="Reading Gamespot Reviews into Dataset")
        for url in gamespot_urls:
            game = url.split("/")[2]
            if game not in visited_gamespot:
                myurl = base + url
                if GamespotWebScraper.scrape_gamespot_review(myurl, counter):
                    counter += 1
                visited_gamespot.add(game)
            progress_bar.update(1)
        progress_bar.close()
        
        print("Starting Destructoid Reviews")
        visited_destructoid = set()
        destructoid_urls = destructoid_urls_f.read()
        destructoid_urls = eval(destructoid_urls)
        progress_bar = tqdm(total=len(destructoid_urls), desc="Reading Destructoid Reviews into Dataset")
        # for url in destructoid_urls:
        for url in destructoid_urls:
            game = url.split("/")[4]
            if game not in visited_destructoid:
                if DestructoidWebScraper.scrape_destructoid_review(url, counter):
                    counter += 1
                visited_destructoid.add(game)
            progress_bar.update(1)
        progress_bar.close()

        print("Starting PCGAMER Reviews")
        visited_pcg = set()
        pc_gamer_urls = pc_gamer_urls_f.read()
        pc_gamer_urls = eval(pc_gamer_urls)
        progress_bar = tqdm(total=len(pc_gamer_urls), desc="Reading PC Gamer Reviews into Dataset")
        for url in pc_gamer_urls:
            game = url.split("/")[3]
            if game not in visited_pcg:
                if PCgamerWebScraper.scrape_pc_gamer_review(url, counter):
                    counter += 1
                visited_pcg.add(game)
            progress_bar.update(1)
        progress_bar.close()

def create_urls():
    GamespotGetURLs.main()
    DestructoidGetURLs.main()
    pcgamer_get_urls.main()
    
def create_games_radar_dataset():
    # get_games_radar_urls.main()
    games_radar_reviews = 'C:/Users/nicho/OneDrive/Desktop/Project/Webscrape/games_radar_urls.txt'
    with open(games_radar_reviews, 'r') as f:
        games_radar_reviews = eval(f.read())

    counter = 1
    print("Starting games_radar Reviews")
    visited_games_radar = set()
    progress_bar = tqdm(total=len(games_radar_reviews), desc="Reading Games Radar Reviews into Dataset")
    for url in games_radar_reviews:
        game = url.split("/")[3]
        if game not in visited_games_radar:
            if gamesradarscraper.scrape_games_radar_review(url, counter):
                counter += 1
            visited_games_radar.add(game)
        progress_bar.update(1)
    progress_bar.close()    


def main():

    # create_games_radar_dataset()

    # create_urls()
    # create_datasets()
    filenames =  ['C:/Users/nicho/OneDrive/Desktop/Project/Data/gamespot_dataset.csv', 'C:/Users/nicho/OneDrive/Desktop/Project/Data/destructoid_dataset.csv', 'C:/Users/nicho/OneDrive/Desktop/Project/Data/pc_gamer_dataset.csv']
    dataframes = []
    for filename in filenames:
        dataframes.append(pd.read_csv(filename))

    full_dataset = pd.concat(dataframes, axis=0, ignore_index=True)
    full_dataset.to_csv('C:/Users/nicho/OneDrive/Desktop/Project/Data/dataset.csv')

if __name__ == "__main__":
    main() 