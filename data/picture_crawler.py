import requests
from bs4 import BeautifulSoup
import os
import traceback


def download_picture(url, filename):
    if os.path.exists(filename):
        print(f"File already existed.\nURL: {url}\n")
        return False
    else:
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            return True
        except Exception:
            traceback.print_exc()
            if os.path.exists(filename):
                os.remove(filename)
            return False


def crawl_konachan(path, picture_limit=20):
    if not os.path.exists(path):
        os.makedirs(path)
    
    url_base = "https://konachan.com"
    picture_count = 0
    page = 1
    while True:
        main_url = f"{url_base}/post?page={page}&tags=rating:safe"
        main_html = requests.get(main_url).text
        main_soup = BeautifulSoup(main_html, "html.parser")

        for img_page in main_soup.find_all("a", class_="thumb"):
            img_page_url = f"{url_base}{img_page['href']}"
            img_page_html = requests.get(img_page_url).text
            img_page_soup =  BeautifulSoup(img_page_html, "html.parser")
            
            img = img_page_soup.find("div", id="right-col").find("img")["src"]

            filename = os.path.join(path, img.split("-")[-1].replace("%20", ""))
            if download_picture(img, filename):
                picture_count += 1
                print(f"[{picture_count} / {picture_limit}] download: {filename}.", end="\r")
                if picture_count >= picture_limit:
                    print(f"\n[{picture_count} / {picture_limit}] success.")
                    return
        page += 1


crawl_konachan("path")
