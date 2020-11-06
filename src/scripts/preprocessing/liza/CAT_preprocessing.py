# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:48:13 2020
@author: lizan
"""
###Очистка ппредназначена для текстов статей, извлечённых из pdf с помощью PDFtotext

import os
import re
from collections import Counter

def clean_text(text):
    n = len(text)
    text2 = remove_begining_text(text)
    begining_removed = False
    if len(text2) < n:
        begining_removed = True
    ending_removed = False
    n = len(text2)
    text2 = remove_ending_text(text2)
    if len(text2) < n:
        ending_removed = True
    paragraphs = text2.split('\n\n')
    #print('Изначально', len(paragraphs))
    paragraphs = remove_page_headers(paragraphs)
    paragraphs = remove_joined_page_headers(paragraphs)
    paragraphs = restore_paragraph_bondaries(paragraphs)
    #print('Колонтитулы удалены', len(paragraphs))
    if not begining_removed:
        paragraphs = remove_begining_paragraphs(paragraphs)
        #print(type(paragraphs))
        #print('Введение удалено', len(paragraphs))
    if not ending_removed:
        paragraphs = remove_ending_paragraphs(paragraphs)
        #print('Конец удален', len(paragraphs))
    paragraphs = choose_russian_paragraphs(paragraphs)
    #print('Остались только абзацы на русском', len(paragraphs))
    paragraphs = remove_short_lines_paragraphs(paragraphs)
    #print('Попытка удалить таблицы', len(paragraphs))
    paragraphs_no_line_bondaries = [re.sub('\n', ' ', paragraph) for paragraph in paragraphs]
    new_text = '\n\n'.join(paragraphs_no_line_bondaries)
    new_text = remove_quotation(new_text)
    new_text = remove_bad_sentences(new_text)
    new_text = remove_references(new_text)
    new_text = remove_url(new_text)
    new_text = re.sub('[ ]{2,}', ' ', new_text)
    new_text = re.sub('[\s]{2,}', '\n\n', new_text)
    new_text = re.sub('[ñâÿçàíšãóìøþôõýæùúñî¹áîřðûɫɥɨêɜǎäöüéïïèåòè►●]', '', new_text)
    new_text = re.sub('[«»“”""„“]', '"', new_text)
    return new_text


##Находит и удаляет одинаковые абзацы (возможно, с разными числами)
def remove_page_headers(paragraphs):
    paragraphs_no_digits = [re.sub('[0-9]+', 'NUM', paragraph) for paragraph in paragraphs]
    page_headers = set()
    known_paragraphs = set()
    for paragraph in paragraphs_no_digits:
        if paragraph in known_paragraphs:
            page_headers.add(paragraph)
        else:
            known_paragraphs.add(paragraph)
    new_paragraphs = []
    if page_headers:
        for i in range(len(paragraphs)):
            if paragraphs_no_digits[i] not in page_headers:
                new_paragraphs.append(paragraphs[i])
    else:
        new_paragraphs = paragraphs
    return new_paragraphs

##Находит и удаляет одинаковые первые/последние строки абзацев (возможно, с разными числами)
def remove_joined_page_headers(paragraphs):
    known_beginings = set()
    known_endings = set()
    ph_candidates = []
    double_paragraphs = [[p, re.sub('[0-9]+', 'Ч', p)] for p in paragraphs]
    double_paragraphs_as_lines = [[p[0].split('\n'), p[1].split('\n')] for p in double_paragraphs]
    for double_paragraph in double_paragraphs_as_lines:
        lines_no_digits = double_paragraph[1]
        if len(lines_no_digits) > 1 and type(lines_no_digits) == list:
            begining = lines_no_digits[0]
            ending = lines_no_digits[-1]
            if begining in known_beginings:
                ph_candidates.append(begining)
            else:
                known_beginings.add(begining)
            if ending in known_endings:
                ph_candidates.append(ending)
            else:
                known_endings.add(ending)
    ph_counter = Counter(ph_candidates)
    page_headers = set()
    for ph_candidate in set(ph_candidates):
        if ph_counter[ph_candidate] > 2:
            page_headers.add(ph_candidate)
    new_paragraphs = []
    for i in range(len(double_paragraphs_as_lines)):
        changes = []
        p = double_paragraphs_as_lines[i]
        if type(p[1]) == list and len(p[1]) > 1:
            if p[1][0] in ph_candidates and len(re.sub('[Ч\s]', '', p[1][0])) > 3:
                changes.append('begining')
            if p[1][1] in ph_candidates and len(re.sub('[Ч\s]', '', p[1][1])) > 3:
                changes.append('ending')
        if not changes:
            new_paragraph = paragraphs[i]
        else:
            new_lines = []
            if 'begining' not in changes:
                new_lines.append(p[0][0])
            if len(p[0]) > 2:
                new_lines += p[0][1:-1]
            if 'ending' not in changes:
                new_lines.append(p[0][-1])
            new_paragraph = '/n'.join(new_lines)
        new_paragraphs.append(new_paragraph)
    return new_paragraphs
                


## Соединяет абзацы, которые начинаются с маленькой буквы, с предшествующими                
def restore_paragraph_bondaries(paragraphs):
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if type(paragraphs) != list or len(paragraphs) < 2:
        return(paragraphs)
    else:
        new_paragraphs = []
        current_paragraph = paragraphs[0]
        for i in range(1, len(paragraphs)):
            if re.match('[а-яёa-z]', paragraphs[i]):
                if current_paragraph[-1] == '-':
                    current_paragraph = current_paragraph[:-1] + paragraphs[i]
                else:
                    current_paragraph = current_paragraph + ' ' + paragraphs[i]
            else:
                new_paragraphs.append(current_paragraph)
                current_paragraph = paragraphs[i]
        new_paragraphs.append(current_paragraph)
    return(new_paragraphs)
               

def short_strings_found(paragraph_as_list, threshold=15):
    n_found = 0
    too_many = False
    if type(paragraph_as_list) == list and len(paragraph_as_list)>1:
        lines_to_check = paragraph_as_list[:-1]
        for line in lines_to_check:
            if type(line) == str and len(line) < threshold:
                n_found += 1
        if n_found > 0.33*len(lines_to_check):  
            too_many = True
    return too_many
        
### Удаляет абзацы, в которых есть короткие строки не в конце (списки, таблицы)      
def remove_short_lines_paragraphs(paragraphs):
     n = len(paragraphs)
     long_lines_paragraphs = []
     paragraphs_as_lines = [paragraph.split('\n') for paragraph in paragraphs]
     for i in range(len(paragraphs_as_lines)):
         paragraph = paragraphs_as_lines[i]
         if not short_strings_found(paragraph):
                 long_lines_paragraphs.append(paragraphs[i])
     if len(long_lines_paragraphs) > 0.5*n:
         return long_lines_paragraphs
     else:
         print('При удалении таблиц чуть не снесли весь текст')
         return paragraphs
 
#Удаляет цитаты    
def remove_quotation(text):
    n = len(text)
    new_text = re.sub('«[^«»{50,500}]»', '', text)
    if len(new_text) > 0.7*n:
        return new_text
    else:
        return text
    
#Удаляет ссылки в квадратных скобках    
def remove_references(text):
    n = len(text)
    new_text = re.sub('\[(.*?)\]', '', text)
    if len(new_text) < n and len(new_text) > 0.7*len(text):
        return new_text
    else:
        return text    
    
def remove_begining_text(text):
    new_text = re.sub('.*?Вве:дение(\.)?', 'Введение. ', text)
    return new_text

#Используется после разбиения на абзацы для попытки убрать шапку, если не сработала регулярка    
def remove_begining_paragraphs(paragraphs):
    annotation_candidate = -1
    n = len(paragraphs)
    max_begining = min(15, n//4)
    for i in range(max_begining):
        if len(re.findall('[a-zA-Z]+', paragraphs[i])) > 35:
            annotation_candidate = i
    if annotation_candidate > 2:
        new_paragraphs = paragraphs[i+1:]
    else:
        new_paragraphs = paragraphs[5:]
    return new_paragraphs
        
#Удаляет абзацы, в которых преобладает иноязычный текст
def choose_russian_paragraphs(paragraphs):
    new_paragraphs = []
    for paragraph in paragraphs:
        if paragraph:
            if len(re.findall('[а-яА-Я]', paragraph)) > 0.2*len(paragraph):
                new_paragraphs.append(paragraph)
    return new_paragraphs
        
def remove_ending_text(text):
#Удаляет список литературы и последующий текст
   n = len(text)
   new_text = re.sub('(Литература|Использованная литература|Список использованной литературы|Список литературы|Библиография|References|Библиографический список)(\.)?.*', '', text, flags=re.S)
   if len(new_text) > 0.7*n:
       return new_text
   else:
       return text
   
#Используется после разбиения на абзацы для попытки убрать концовку, если не сработала регулярка        
def remove_ending_paragraphs(paragraphs):
    new_paragraphs = paragraphs[:-2]
    return new_paragraphs
    
    
def remove_url(text):
    new_text = re.sub('([^ ])+([@])([^ ]+)', '', text)
    new_text = re.sub('https:[^ ]+', 'URL', new_text)
    new_text = re.sub('[^ ]+?/[^ ]+?/[^ ]+?', 'URL', new_text)
    return new_text
 
#Удаляет предложения, в которых преобладает иноязычный текст
# либо много однобуквенных слов подряд (типографская разрядка)    
def remove_bad_sentences(text):
    sentences = text.split('.')
    new_sentences = []
    for sentence in sentences:
        words = sentence.split()
        words_len = [str(len(word)) for word in words]
        len_describrion = ' '.join(words_len)
        if '1 1 1 1 1' not in len_describrion and len(re.findall('[а-яА-ЯёЁ]', sentence)) > 0.3 * len(sentence):
            new_sentences.append(sentence)
    new_text = '.'.join(new_sentences)
    return new_text
    
def clean_file(file):
    with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    new_text = clean_text(text)
    return new_text
    
def clean_directory(directory):
    os.chdir(directory)
    try:
        os.mkdir('cleaned')
    except:
        pass
    for file in os.listdir(directory):
        if os.path.isfile(file):
            text = clean_file(file)
            txt_name = re.search('[0-9]+', file).group(0)
            if not txt_name:
                txt_name = file
            txt_name = 'cleaned\\' + txt_name +'.txt'
            with open(txt_name, 'w', encoding='utf-8') as f2:
                f2.write(text)
            print(file, 'is cleaned')

if __name__ == '__main__':
    directory = input('Директория с извлеченными txt')
    clean_directory(directory)