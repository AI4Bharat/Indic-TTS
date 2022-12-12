import re
import json

num_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)?")
def get_all_numbers_from_string(text):
  return num_str_regex.findall(text)

from indic_numtowords import num2words, supported_langs
import traceback
from .translator import GoogleTranslator

class TextNormalizer:
  def __init__(self):
    self.translator = GoogleTranslator()
    self.symbols2lang2word = json.load(open('src/utils/symbols.json', 'r'))
  
  def normalize_text(self, text, lang):
    text = self.replace_punctutations(text, lang)
    text = self.convert_symbols_to_words(text, lang)
    text = self.convert_numbers_to_words(text, lang)
    return text

  def replace_punctutations(self, text, lang):
    if lang not in ['brx', 'or']:
      text = text.replace('।', '.')
    else:
      text = text.replace('.', '।')
    text = text.replace('|', '.')
    text = text.replace(':', ',').replace(';',',')
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

  def expand_phones(self, item):
    return ' '.join(list(item))
  
  def convert_symbols_to_words(self, text, lang):
    symbols = self.symbols2lang2word.keys()
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    urls = re.findall(r"\w+://\w+\.\w+\.\w+/?[\w\.\?=#]*", text)
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
