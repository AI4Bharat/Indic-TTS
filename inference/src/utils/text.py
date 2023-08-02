import os
PWD = os.path.dirname(__file__)
import re
import regex
import json
import traceback

from nemo_text_processing.text_normalization.normalize import Normalizer
from indic_numtowords import num2words, supported_langs
from .translator import GoogleTranslator

indic_acronym_matcher = regex.compile(r"([\p{L}\p{M}]+\.\s*){2,}")

# short_form_regex = re.compile(r'\b[A-Z\.]{2,}s?\b')
# def get_shortforms_from_string(text):
#     return short_form_regex.findall(text)

short_form_regex = re.compile(r"\b([A-Z][\.\s]+)+([A-Z])?\b")
eng_consonants_regex = re.compile(r"\b[BCDFGHJKLMNPQRSTVWXZbcdfghjklmnpqrstvwxz]+\b")
def get_shortforms_from_string(text):
  dotted_shortforms = [m.group() for m in re.finditer(short_form_regex, text)]
  non_dotted_shortforms = [m.group() for m in re.finditer(eng_consonants_regex, text)]
  return dotted_shortforms + non_dotted_shortforms

decimal_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)")
def get_all_decimals_from_string(text):
  return decimal_str_regex.findall(text)

num_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)?")
def get_all_numbers_from_string(text):
  return num_str_regex.findall(text)

multiple_stops_regex = r'\.\.+'
def replace_multiple_stops(text):
  return re.sub(multiple_stops_regex, '.', text) 

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

email_regex = r'[\w.+-]+@[\w-]+\.[\w.-]+'
url_regex = r'((?:\w+://)?\w+\.\w+\.\w+/?[\w\.\?=#]*)|(\w*.com/?[\w\.\?=#]*)'
currency_regex = r"\₹\ ?[+-]?[0-9]{1,3}(?:,?[0-9])*(?:\.[0-9]{1,2})?"
phone_regex = r'\+?\d[ \d-]{6,12}\d'



class TextNormalizer:
  def __init__(self):
    self.translator = GoogleTranslator()
    self.normalizer = Normalizer(input_case='cased', lang='en')
    self.symbols2lang2word = json.load(open(os.path.join(PWD, "symbols.json"), encoding="utf-8"))
    self.alphabet2phone = json.load(open(os.path.join(PWD, "alphabet2phone.json"), encoding="utf-8"))
  
  def normalize_text(self, text, lang):
    text = text.replace("।", ".").replace("|", ".").replace("꯫", ".").strip()
    text = self.expand_shortforms(text, lang)
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
    return text   

  def replace_punctutations(self, text, lang):
    text = replace_multiple_stops(text)
    if lang not in ['brx', 'or']:
      text = text.replace('।', '.')
      if text[-1] not in ['.', '!', '?', ',', ':', ';']:
        text = text + ' .'
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
      # print(lang, numbers)
      num_words = [num2words(num, lang=lang) for num in numbers]
    else: # Fallback, converting to Indian-English, followed by NMT
      try:
        num_words = [num2words(num, lang="en") for num in numbers]
        translated_num_words = [self.translator(text=num_word, from_lang="en", to_lang=lang) for num_word in num_words]
        # TODO: Cache the results?
        num_words = translated_num_words
      except:
        traceback.print_exc()
    
    for num_str, num_word in zip(num_strs, num_words):
      text = text.replace(num_str, ' '+num_word+' ', 1)
    return text.replace("  ", ' ')

  def convert_dates_to_words(self, text, lang):
    date_strs = get_all_dates_from_string(text)
    if not date_strs:
      return text
    for date_str in date_strs:
      normalized_str = self.normalizer.normalize(date_str, verbose=False, punct_post_process=True)
      if lang in ['brx', 'en']:  # no translate
        translated_str = normalized_str
      else:
        translated_str = self.translator(text=normalized_str, from_lang="en", to_lang=lang)
      text = text.replace(date_str, translated_str)
    return text

  def expand_phones(self, item):
    return ' '.join(list(item))
  
  def find_valid(self, regex_str, text):
    items = re.findall(regex_str, text)
    return_items = []
    for item in items:
      if isinstance(item, tuple):
        for subitem in item:
          if len(subitem) > 0:
            return_items.append(subitem)
            break  # choose first valid sub item
      elif len(item) > 0:
        return_items.append(item)
    return return_items
  
  def convert_symbols_to_words(self, text, lang):
    symbols = self.symbols2lang2word.keys()
    emails = self.find_valid(email_regex, text)
    # urls = re.findall(r'(?:\w+://)?\w+\.\w+\.\w+/?[\w\.\?=#]*', text)
    urls = self.find_valid(url_regex, text)
    # print('URLS', urls)
    for item in emails + urls:
      item_norm = item
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      text = text.replace(item, item_norm)
    
    currencies = self.find_valid(currency_regex, text)
    for item in currencies:
      item_norm = item.replace('₹','') + '₹'  # Pronounce after numerals
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      text = text.replace(item, item_norm)
    
    phones = self.find_valid(phone_regex, text)
    for item in phones:
      item_norm = item.replace('-', ' ')
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      item_norm = self.expand_phones(item_norm)
      text = text.replace(item, item_norm)
    
    # percentage
    text = text.replace('%', self.symbols2lang2word['%'][lang])
    
    return text

  def convert_char2phone(self, char):
        return self.alphabet2phone[char.lower()] if char.lower() in self.alphabet2phone else ''
  
  def expand_shortforms(self, text, lang):
    if lang!='en':
      # Remove dots, as it speaks out like each letter is separate sentence
      # Example: अई. अई. टी. -> अई अई टी
      for match in regex.finditer(indic_acronym_matcher, text):
        match = match.group()
        match_without_dot = match.replace('.', ' ')
        text = text.replace(match, match_without_dot)
      return text
    
    shortforms = get_shortforms_from_string(text)
    for shortform in shortforms:
        shortform = shortform.strip()
        if shortform == 'I' or shortform == "A":
          # Skip if valid English words
          continue
        expanded = ' '.join([self.convert_char2phone(char) for char in shortform])
        text = text.replace(shortform, expanded, 1)
    return text
