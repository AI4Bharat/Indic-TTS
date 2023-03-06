import os
PWD = os.path.dirname(__file__)

import re
import json
from nemo_text_processing.text_normalization.normalize import Normalizer

decimal_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)")
def get_all_decimals_from_string(text):
  return decimal_str_regex.findall(text)

num_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)?")
def get_all_numbers_from_string(text):
  return num_str_regex.findall(text)

date_generic_match_regex = re.compile("(?:[^0-9]\d*[./-]\d*[./-]\d*)")
date_str_regex = re.compile("(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4})|(?:\d{2,4}[./-]\d{1,2}[./-]\d{1,2})")  # match like dd/mm/yyyy or dd-mm-yy or yyyy.mm.dd or yy/mm/dd
def get_all_dates_from_string(text):
  candidates = date_generic_match_regex.findall(text)
  candidates = [c.replace(' ', '') for c in candidates]
  candidates = [c for c in candidates if len(c) <= 10]  # Prune invalid dates
  candidates = ' '.join(candidates)
  return date_str_regex.findall(candidates)

def get_decimal_substitution(decimal):
  decimal_parts = decimal.split('.')
  l_part = decimal_parts[0]
  r_part = ""
  for part in decimal_parts[1:]:
    r_part += ' '.join(list(part))  # space between every digit after decimal point
  decimal_sub = l_part + " point " + r_part 
  decimal_sub = decimal_sub.strip()
  return decimal_sub


from indic_numtowords import num2words, supported_langs
import traceback
from .translator import GoogleTranslator

class TextNormalizer:
  def __init__(self):
    self.translator = GoogleTranslator()
    self.normalizer = Normalizer(input_case='cased', lang='en')
    self.symbols2lang2word = json.load(open(os.path.join(PWD, "symbols.json"), encoding="utf-8"))
  
  def normalize_text(self, text, lang):
    text = self.normalize_decimals(text, lang)
    text = self.replace_punctutations(text, lang)
    text = self.convert_dates_to_words(text, lang)
    text = self.convert_symbols_to_words(text, lang)
    text = self.convert_numbers_to_words(text, lang)
    return text
  
  def normalize_decimals(self, text, lang):
    decimal_strs = get_all_decimals_from_string(text)
    if not decimal_strs:
      return text
    decimals = [str(decimal_str.replace(',', '')) for decimal_str in decimal_strs]
    decimal_substitutions = [get_decimal_substitution(decimal) for decimal in decimals]
    for decimal_str, decimal_sub in zip(decimal_strs, decimal_substitutions):
      text = text.replace(decimal_str, decimal_sub)
    print("normalized_decimals", text)
    return text   


  def replace_punctutations(self, text, lang):
    if lang not in ['brx', 'or']:
      text = text.replace('।', '.')
    else:
      text = text.replace('.', '।')
    text = text.replace('|', '.')
    for bracket in ['(', ')', '{', '}', '[', ']']:
      text = text.replace(bracket, ',')
    # text = text.replace(':', ',').replace(';',',')
    text = text.replace(';',',')
    return text
  
  def convert_numbers_to_words(self, text, lang):
    num_strs = get_all_numbers_from_string(text)
    if not num_strs:
      return text
    
    # TODO: If it is a large integer without commas (say >5 digits), spell it out numeral by numeral
    # NOTE: partially handled by phones
    numbers = [int(num_str.replace(',', '')) for num_str in num_strs]
    
    if lang in supported_langs:
      print(lang, numbers)
      num_words = [num2words(num, lang=lang) for num in numbers]
    else: # Fallback, converting to Indian-English, followed by NMT
      try:
        num_words = [num2words(num, lang="en") for num in numbers]
        translated_num_words = [self.translator(text=num_word, from_lang="en", to_lang=lang) for num_word in num_words]
        # TODO: Cache the results?
        num_words = translated_num_words
      except:
        traceback.print_exc()
  
    print('TRANSLATED: ', num_words)
    
    for num_str, num_word in zip(num_strs, num_words):
      text = text.replace(num_str, ' '+num_word+' ', 1)
    return text.replace("  ", ' ')

  def convert_dates_to_words(self, text, lang):
    date_strs = get_all_dates_from_string(text)
    print('DATES:', date_strs)
    if not date_strs:
      return text
    for date_str in date_strs:
      normalized_str = self.normalizer.normalize(date_str, verbose=False, punct_post_process=True)
      if 'lang' == 'brx':  # no translate
        translated_str = normalized_str
      else:
        translated_str = self.translator(text=normalized_str, from_lang="en", to_lang=lang)
      print('translated date:', translated_str)
      text = text.replace(date_str, translated_str)
    return text

  def expand_phones(self, item):
    return ' '.join(list(item))
  
  def convert_symbols_to_words(self, text, lang):
    symbols = self.symbols2lang2word.keys()
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    urls = re.findall(r'(?:\w+://)?\w+\.\w+\.\w+/?[\w\.\?=#]*', text)
    for item in emails + urls:
      item_norm = item
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      text = text.replace(item, item_norm)
    currencies = re.findall(r"\₹\ ?[+-]?[0-9]{1,3}(?:,?[0-9])*(?:\.[0-9]{1,2})?", text)
    for item in currencies:
      item_norm = item.replace('₹','') + '₹'  # Pronounce after numerals
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      text = text.replace(item, item_norm)
    phones = re.findall(r'\+?\d[ \d-]{6,12}\d', text)
    for item in phones:
      item_norm = item.replace('-', ' ')
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      item_norm = self.expand_phones(item_norm)
      text = text.replace(item, item_norm)
    return text
