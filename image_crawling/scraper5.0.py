from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from urllib.request import urlretrieve, urlopen, Request
#pip install chromedriver-autoinstaller
import chromedriver_autoinstaller
#pip install webdriver-auto-update
import glob
import os
import csv
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from selenium.webdriver.common.action_chains import ActionChains
import requests
import cv2
from selenium.webdriver.support.ui import WebDriverWait
from datetime import datetime
now = datetime.now()


#navigating through results




class ChromefoxTest:

    

    def __init__(self,url):
        self.url=url
        self.uri = []
        
        if (not os.path.exists(folder)):
            os.makedirs(folder)
            
        if not os.path.exists(polarities_dir):
            os.makedirs(polarities_dir) 
            
        if os.path.exists(polarities_dir+"/.DS_Store"):
            os.remove(polarities_dir+"/.DS_Store")




    

    def chromeTest(self):

        

        chrome_options = Options()
        #chrome_options.add_argument("--headless")   #no windows -> run as background simulation
        chrome_options.add_argument("--window-size=1920,1080")  #instead of full screen -> solves less pictures issue
        chrome_options.add_argument("--start-maximized")    #maximize window
        chrome_options.add_argument("--disable-gpu")        #better performance for background scripts
        chrome_options.add_argument("--disable-extensions") 
        chrome_options.experimental_options["useAutomationExtension"] = False 
        chrome_options.add_argument('--ignore-certificate-errors')  #otherwise blank pages sometimes
        chrome_options.add_argument('--allow-running-insecure-content')  
        chrome_options.add_argument("--proxy-server='direct://'")
        chrome_options.add_argument("--proxy-bypass-list=*")
        chrome_options.add_argument("--disable-notifications") #new
        chrome_options.add_argument("user-data-dir=./images/persona/"+persona) 

        self.number_same_images = 0
        self.number_multiple_same_images = 0
        self.number_new_images = 0

        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        SCROLL_PAUSE_TIME = random.randint(2, 4)  #++random.rand(0-3) because of many queries, 

         

        new_height = 0
        #setCookies(persona, self.driver)
        self.driver.get(self.url)  
        
        buttons = self.driver.find_elements_by_tag_name("button")

        if len(buttons) > 1:
            for button in buttons:
                if button.text == 'Accept all' or 'Accept':
                    button.click()
                    break

        last_height = self.driver.execute_script("return document.body.scrollHeight") #split to many scrolls
        while True:
            # Scroll down to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)
            
            # Calculate new scroll height and compare with last scroll height
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:

                    if search_engine == 'Google':
                        self.driver.find_element_by_class_name('mye4qd').click()    #class name of "Load more" button
                        time.sleep(SCROLL_PAUSE_TIME)
                        pass

                    elif search_engine == 'Bing':
                        self.driver.find_element_by_class_name('btn_seemore.cbtn.mBtn').click()
                        time.sleep(SCROLL_PAUSE_TIME)

                except:
                    break


            last_height = new_height
            """
            element = self.driver.find_element_by_class_name('isv-r.PNCib.MSM1fd.BUooTd')
            element.click()

            element_preview = self.driver.find_element_by_class_name('n3VNCb.KAlRDb')
            src = element_preview.get_attribute("src")"""

            if search_engine == 'Google':
                self.elements_preview = self.driver.find_elements_by_class_name('isv-r.PNCib.MSM1fd.BUooTd')
            elif search_engine == 'Bing':
                self.elements_preview = self.driver.find_elements_by_class_name('mimg')    
        

            
        
    
        #self.driver.get_screenshot_as_file("error_" + str(now.strftime("%d-%m-%Y_%H:%M:%S")) + ".png")    #debug

        broken = False
        for v in self.elements_preview:

            if broken == True:
                break

            
            actions = ActionChains(self.driver)
            actions.move_to_element(v).perform()    #scroll to image high


            if search_engine == 'Google':
                
            
                v.click()
                time.sleep(3)
                #element_preview = self.driver.find_element_by_class_name('n3VNCb.KAlRDb') #sometimes just n3VNCb
                #element_preview = self.driver.find_elements(By.XPATH("//[contains(@id, 'n3VNCb')]"))


                elements = self.driver.find_elements_by_css_selector("img[class*='n3VNCb']")

            elif search_engine == 'Bing':

                preview_img = v.get_attribute("src")
                elements = self.driver.find_elements_by_css_selector("a[class*='iusc']")



            for element in elements[1:]:    #skip first, otherwise it will lead to too many overrides, see documentations

                headers = {
                      #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; Windows NT 6.1; Win64; x64) ' 
                      'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) ' 
                      'AppleWebKit/537.11 (KHTML, like Gecko) '
                      'Chrome/23.0.1271.64 Safari/537.11',
                      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                      'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                      'Accept-Encoding': 'none',
                      'Accept-Language': 'en-US,en;q=0.8',
                      'Connection': 'keep-alive'}

                time.sleep(random.randint(1, 3))

                if search_engine == 'Google':
                    src = element.get_attribute("src")
                    req = Request(url=src, headers=headers)
                    try:
                        resource = urlopen(req, timeout=10)
                        output = open(folder + "/raw.jpg","wb")
                        output.write(resource.read())
                        output.close()
                        new_name = fileHasher(folder + "/raw.jpg")
                        os.rename(folder + "/raw.jpg", folder + "/" + new_name)
                        h, w, c = cv2.imread(folder + "/" + new_name).shape       
                        dublicate = checkDuplicate(self, new_name, h, w)
                    except:
                        break

                elif search_engine == 'Bing':
                    src = json.loads(element.get_attribute("m"))        #has a json format 
                    full_size_img = src['murl']
                    req = Request(url=full_size_img, headers=headers)
                
                    try:

                        resource = urlopen(req, timeout=10)
                        output = open(folder + "/raw.jpg","wb")
                        output.write(resource.read())
                        output.close()
                        new_name = fileHasher(folder + "/raw.jpg")
                        os.rename(folder + "/raw.jpg", folder + "/" + new_name)
                        h, w, c = cv2.imread(folder + "/" + new_name).shape        #nonetyoe object appears out of nowhere in the middle
                        dublicate = checkDuplicate(self, new_name, h, w)                            
                    except:
                        pass

                

                #width = w
                #height = h
             
                """
                if self.number_new_images >= 10:
                    broken = True
                    break
                """


                
                """------------------------------------------------------------ Very Small pictures                                             #plot all images and check their dimensions, groupby, extract information like filetype (not extension, first to bytes defines filetype, "detect filetyp python")
                if (width * height <= 80*80):          #very small images are usually not advertisements. False positives drops by 24% 
                    small_folder = self.folder + '/very small'

                    if ((not os.path.exists(small_folder))):            #if very small images are found, save them into a extra directory
                        os.makedirs(small_folder)
                    
                    os.rename(self.folder + "/" + new_name, small_folder + "/" + new_name)"""


            
        


          #classes
        directories = [path_training + polarities[0], path_training + polarities[1],        #all directorys (directory + class) 
                            path_testing + polarities[0], path_testing + polarities[1]]


        plotNumber(self)
        writeRecords(self)

        theSplitter(topic, polarity, query, directories, path_training, path_testing, polarities)


        
def fileHasher(file):
        import hashlib
        with open(file, 'rb') as opened_file:
            content = opened_file.read()
            sha256 = hashlib.sha256()                       #options are md5, sha1, sha224, sha256, sha384, sha512

            sha256.update(content)
            hashValue = sha256.hexdigest()
        return hashValue

#bash imagescript, extract for image attributes, -> decision tree


def checkDuplicate(self, file, h, w):

    
    
    paths = []

    for polar in polarities:                                                #check for all directories in topic and safe them to paths

        try:
            queries = os.listdir(polarities_dir+polar+"/")
  
            
        
            ds_store = queries.count(".DS_Store")
            if ds_store > 0:
                queries.remove(".DS_Store")  #.DS_Store hidden directory
            if polar == polarity:              #if the polarity in the loop is the polarity of our image = own folder for the image
                queries.remove(query)       #don't search for dublicates in own folder, variables are for current path")


            for querie in queries:              
                paths.append(polarities_dir+polar+"/"+querie)   

        except:
            print(polarities_dir+polar+"/ - Does not exist yet")

    compare_file = ""
    file_paths = []
    file_queries = []
    file_count = 0
    first_time = 1
 
    for path in paths:
        
        for files in os.walk(path, topdown=False):
            
            for name in files[2]:       #file[0] is path, file[1] is empty, file[2] stores all names of all files 
                if name == file:

                    if compare_file != file:                                        #prevents double counting if file is already found in one directory and again in another directory
                        file_count = file_count + 1
                        print("Already exists in: " + path)
                        file_paths.append(path)
                        file_query = path.split('/')[-1]        #get query = last element from path
                        file_queries.append(file_query)

                        self.number_same_images = self.number_same_images + 1       #does not include images in same directory
                    
                    else:
                        if first_time == 1:         #cupdate variable only once and not per directory
                             
                            
                            self.number_same_images = self.number_same_images - 1       
                            self.number_multiple_same_images = self.number_multiple_same_images + 1 
                            file_count = file_count + 1
                            first_time = 0          

                        file_paths.append(path)

                       
                        file_query = path.split('/')[-1]        #get query = last element from path
                        file_queries.append(file_query)

                        print("This picture exists in multiple directories.")                             #in this case a image is already saved more than once
                        print("Writing all informations about that file in a csv...")        
                            
        

                    compare_file = file
                    break
    
    if file_count == 0:                     #if its not a dublicate 
        file_count = file_count + 1
        file_queries = query
        file_paths = folder
        self.number_new_images = self.number_new_images + 1       
           
    fileindexWriter(file, file_count, file_queries, file_paths, h, w)


    


def fileindexWriter(compare_file, file_count, file_queries, file_paths, h, w):

    header = ['image', 'count', 'topic', 'polarity', 'query', 'directories', 'height', 'width']
    data = [compare_file, file_count, topic, polarity, file_queries, file_paths, h, w]

    with open(polarities_dir + 'file_index'+ str(now.strftime("%d-%m-%Y_%H:%M:%S")) + '.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            if csv_file.tell() == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)

            csv_file.close()


def theSplitter(topic, polarity, query, directories, path_training, path_testing, polarities):
    percentage = 30         #max 100
  




    for directory in directories:               #check if all directories in array exists and if not create it
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        


    while(True): 
        import shutil, random, os
            
        
        files = os.listdir(folder)
 


        for f in files:
            


            if (f == "very small"):          #directory "very small should not be moved"
                pass
            elif polarity == 'positive':
                shutil.copyfile(folder + '/' + f, path_training + polarities[0] + '/' + f)
            else:
                shutil.copyfile(folder + '/' + f, path_training + polarities[1] + '/' + f)
        
        try: 
            percentage = round((percentage/100) * len(files))
            randomfiles = random.sample(os.listdir(path_training + polarities[0]), percentage)     # select random files for percentage 
            for fname in randomfiles:
                srcpath = path_training + polarities[0] + '/' + fname
                destPath = os.path.join(path_testing + polarities[0] + '/', fname)
                shutil.move(srcpath, destPath)

            randomfiles_2 = random.sample(os.listdir(path_training + polarities[1]), percentage)     # select random files for percentage 
            for fname in randomfiles_2:
                srcpath = path_training + polarities[1] + '/' + fname
                destPath = os.path.join(path_testing + polarities[1] + '/', fname)
                shutil.move(srcpath, destPath)
        except:
            break

        break  

    #plotImages(directories)

    



 #------------------------------------------------------------------------------------------------------       
def plotImages(directories, file):
    classes = ['very small', 'small', 'medium', 'big', 'huge']

    files_total = 0

    width_array = []
    height_array = []
    sizes = np.array([])              #very small, small, medium, big, huge


    very_small_counter = 0
    for file in files_total:      #go through all files, extract their size and append them into arrays

        
        for directory in directories:
            try:
                image = PIL.Image.open(directory + '/' + file)
                width, height = image.size
                width_array.append(width)
                height_array.append(height)

                if width * height <= 80 * 80:                       #Should never happen because we pass it in def chrometest and save it in a different directory
                    very_small_counter = very_small_counter+1
                    #image.show()
                    print("Very small: " + very_small_counter)
            except:
                pass

    for x in range(len(width_array)):           #go through all sizes inside the arrays and seperate them into predefined sizes

        if (width_array[x] * height_array[x] <= 80 * 80):              #Should never happen because we pass it in def chrometest and save it in a different directory
            sizes = np.append(sizes, 'very small')

        elif (width_array[x] * height_array[x] <= 300 * 250):
            sizes = np.append(sizes, 'small')
        elif (width_array[x] * height_array[x] <= 500 * 400):
            sizes = np.append(sizes, 'medium')
        elif (width_array[x] * height_array[x] <= 800 * 600):
            sizes = np.append(sizes, 'big')
        else:
            sizes = np.append(sizes, 'huge')
    




    num_very_small = (sizes == 'very small').sum()
    num_small = (sizes == 'small').sum()
    num_medium = (sizes == 'medium').sum()
    num_big = (sizes == 'big').sum()
    num_huge = (sizes == 'huge').sum()

    num_counts = [num_very_small, num_small, num_medium, num_big, num_huge]
 
    x_values = np.arange(1, len(classes) + 1, 1)

    plt.bar(x_values, num_counts, align='center')
    # Decide which ticks to replace.
    new_ticks = ["word for " + str(y) if y != 0.3 else str(y) for y in num_counts]
    plt.yticks(num_counts, new_ticks)
    plt.xticks(x_values, classes)
    plt.savefig(topic + "_images.jpg")
    
#--------------------------------------------------------------------------------------------------




def setCookies(persona, driver):
    import pickle
    




    train = False


    if (train == True):
        with open(trainings_list_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                time.sleep(4)
                driver.implicitly_wait(10)

                driver.get("http://"+line)
                line.replace("\n","")
                
                #WebDriverWait(driver, 100).until()
                print("Adding cookies from"+ line)
                if not os.path.exists(cookiepath):
                    os.makedirs(cookiepath)
                pickle.dump(driver.get_cookies() , open(cookiepath+line.replace("\n","")+".pkl","ab"))
    else:
        files = os.listdir(cookiepath)
        for file in files:
            if file == ".DS_Store":
                os.remove(cookiepath+file)
                break
            cookies = pickle.load(open(cookiepath+file, "rb"))
            file = file.replace(".pkl", "")
            driver.get("http://"+file)
            for cookie in cookies:
                try:
                    driver.add_cookie(cookie)
                except:
                    print("Ups")
           
def page_has_loaded(driver):
    page_state = driver.execute_script('return document.readyState;')
    return page_state == 'complete'

def plotNumber(self):
    langs = ['New images', 'Simple Dublicates', 'Multiple Duplicates']
    numbers = [self.number_new_images, self.number_same_images, self.number_multiple_same_images]
    plt.bar(langs,numbers)
    plt.title('Distribution of images')
    plt.savefig(polarities_dir + "image_distribution_"+ str(now.strftime("%d-%m-%Y_%H:%M:%S")) + ".jpg")
    plt.show()    


def plotPersonas(number_new_images, number_same_images):
    numbers = number_new_images + number_same_images

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    langs = []
    numbers = []

    for persona in ...:

        langs.append(persona)
        numbers.append(persona)


    ax.bar(langs, ...)
    plt.show()    
  
    
def writeRecords(self):

    header = ['search engine', 'persona', 'topic', 'polarity', 'query', 'new images', 'simple duplicates', 'multiple duplicates', 'total images']
    data = [search_engine, persona, topic, polarity, query, self.number_new_images, self.number_same_images, self.number_multiple_duplicates, (self.number_new_images+self.number_same_images+self.number_multiple_duplicates)]

    with open(polarities_dir + 'records_'+str(now.strftime("%d-%m-%Y_%H:%M:%S"))+'.csv', 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            if csv_file.tell() == 0:
                csv_writer.writerow(header)
            csv_writer.writerow(data)

            csv_file.close()



    #https://www.selenium.dev/documentation/webdriver/browser/cookies/
    #https://www.w3.org/TR/webdriver1/#cookies


   


if __name__=='__main__':

        chromedriver_autoinstaller.install()

        with open('images/queries.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:  #row = column, line_count == zeile‚

                    topic, polarity, query = row[0].split(';')
                    
                    #plot images got from different personas
                    #plot images got from different queues
                    #plot images got different topics
                    #plot images got from different polarities

                    search_engine = "Bing"    #Google, Bing
                    persona = "education"       #education, financial, gambling, gaming, health, shopping, travel
                    persona_dir = "images/images/"+ search_engine + "/persona/"
                    polarities_dir = ("images/images/" +search_engine+ "/" +persona+"/"+topic+"/")  



                    
                    polarities = []
                    for row in csv_reader:              #find all polarities in queries.csv and add each unique in array

                        t, p, q = row[0].split(';')

                        if (polarities.count(p) == 0):       
                            polarities.append(p)
                        

                    folder = polarities_dir+polarity+'/'+query #read csv - topic - positiv, negative, querie (might contain commas)  
                    trainings_list_path = "images/training_lists/"+ persona+ "_persona.txt"
                    cookiepath = persona_dir+persona+"/cookies/"

                    path_training = "data/" + topic + "/training/"        
                    path_testing = "data/" + topic + "/testing/" 
                    #polarities = ['for', 'against']   
                    
                    
                    if search_engine == 'Google':
                        FT=ChromefoxTest("https://www.google.com/search?q="+query+"&source=lnms&tbm=isch")
                        FT.chromeTest()

                    elif search_engine == 'Bing':
                        FT=ChromefoxTest("https://www.bing.com/images/search?q="+query+"&form=HDRSC2&first=1&tsc=ImageHoverTitle")
                        FT.chromeTest()

                    line_count += 1
          

        


#DeprecationWarning: find_elements_by_tag_name is deprecated. Please use find_elements(by=By.TAG_NAME, value=name) instead   self.r=self.driver.find_elements_by_tag_name('img')             
            





