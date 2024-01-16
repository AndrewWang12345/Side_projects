import io
from PIL import Image
import requests
import undetected_chromedriver as uc
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium import webdriver as wd
email = 'my email here'
password = 'my password here'
#PATH = 'C:/Users/andrew/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe'
webdriver = uc.Chrome()

def get_google_images(webdriver, max_images):
    def scroll_down(webdriver):
        webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        WebDriverWait(webdriver, 100).until(EC.presence_of_element_located((By.CLASS_NAME, 'Q4LuWd')))
    url = 'google_images url here'
    webdriver.get(url)
    image_urls = set()


    while len(image_urls) < max_images:
        scroll_down(webdriver)
        thumbnails = webdriver.find_elements(By.CLASS_NAME, 'Q4LuWd')
        print(len(thumbnails))
        for img in thumbnails[len(image_urls): max_images]:
            try:
                pic = WebDriverWait(webdriver, 100).until(EC.element_to_be_clickable(img))
                pic.click()
                #time.sleep(1)
            except:
                print("click failed")
                continue
            WebDriverWait(webdriver, 100).until(EC.presence_of_element_located((By.XPATH, "//*[contains(@class, 'sFlh5c pT0Scc iPVvYb')]")))
            images = webdriver.find_elements(By.XPATH, "//*[contains(@class, 'sFlh5c pT0Scc iPVvYb')]")
            #print((images))
            for image in images:
                if image.get_attribute('src') and 'http' in image.get_attribute('src'):
                    image_urls.add(image.get_attribute('src'))
                    print("image found")
        print(len(image_urls))
    return image_urls
def download_image(download_path, url, file_name):
    try:
        image_content = requests.get(url).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file)
        file_path = download_path + file_name
        with open(file_path, 'wb') as f:
            image.save(f, 'JPEG')
        print('success')
    except Exception as e:
        print('failed -', e)
urls = get_google_images(webdriver, 100)
for i, url in enumerate(urls):
    download_image('download path', url, str(i)+'.jpg')
print(urls)
webdriver.quit()