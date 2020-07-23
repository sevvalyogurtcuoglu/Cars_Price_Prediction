# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:05:20 2020

@author: TOSHIBA
"""

from selenium import webdriver
import random
import csv
kilometre = []
model = []
yil = []
price = []
marka = []
seri = []
baslik = []

browser = webdriver.Firefox()

url="*******************"
for i in range(1,12):
    newUrl = url+str(i*20)
    browser.get(newUrl)
    trCount=1
    modell=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[2]')
    markaa=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[3]')
    serii=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[4]')
    baslikk=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[5]')
    yill=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[6]')
    kmm=browser.find_elements_by_xpath('//*[@id="searchResultsTable"]/tbody/tr["trCaunt"]/td[7]')
    fiyat=browser.find_elements_by_css_selector(".searchResultsPriceValue")
    
    for models in modell:
        model.append(models.text) 

    for markas in markaa:
        marka.append(markas.text)  

    for seris in serii:
        seri.append(seris.text) 

    for basliks in baslikk:
        baslik.append(basliks.text) 

    for yils in yill:
        yil.append(yils.text) 

    for kms in kmm:
        kilometre.append(kms.text) 
        #tdCaunt+=1
    
    for fiyats in fiyat:
        price.append(fiyats.text)    
        trCount+=1


browser.close()      
#%%
yakit = []
vites = []

for c in range(0,220):
    yakit.append('Benzin')

for c in range(0,220):
    vites.append('Otomatik')    
#%%

import pandas as pd
df_yeni = pd.DataFrame({'Marka':model,'Model':marka,'Seri':seri,'Başlik':baslik,'Yakit':yakit,'Vites':vites,'Kilometre':kilometre,'Yil':yil,'Price':price}) 

df_yeni.to_csv('tam_arac_tahmin.csv', index=False, encoding='utf-8')
