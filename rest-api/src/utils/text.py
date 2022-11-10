import re
import json

num_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3}){1,3}|(?:\d{1,7}))?(?:\.\d+)?")
def get_all_numbers_from_string(text):
  return num_str_regex.findall(text)

from num2words import num2words
import traceback
from .translator import GoogleTranslator

class TextNormalizer:
  def __init__(self):
    self.translator = GoogleTranslator()
    self.symbols2lang2word = json.load(open('src/utils/symbols.json', 'r'))
  
  def normalize_text(self, text, lang):
    text = self.convert_symbols_to_words(text, lang)
    text = self.convert_numbers_to_words(text, lang)
    return text

  def convert_numbers_to_words(self, text, lang):
    num_strs = get_all_numbers_from_string(text)
    if not num_strs:
      return text
    
    # TODO: If it is a large integer without commas (say >5 digits), spell it out numeral by numeral
    numbers = [float(num_str.replace(',', '')) for num_str in num_strs]

    # TODO: Use this library once stable: https://github.com/sutariyaraj/indic-num2words
    # Currently, we are first converting to Indian-English, followed by NMT
    num_words = [num2words(num, lang="en_IN") for num in numbers]
    if lang != "en":
      try:
        translated_num_words = [self.translator(text=num_word, from_lang="en", to_lang=lang) for num_word in num_words]
        # TODO: Cache the results?
        num_words = translated_num_words
      except:
        traceback.print_exc()
    
    for num_str, num_word in zip(num_strs, num_words):
      text = text.replace(num_str, ' '+num_word+' ', 1)
    return text.replace("  ", ' ')

  def convert_symbols_to_words(self, text, lang):
    symbols = self.symbols2lang2word.keys()
    emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    urls = re.findall(r"\w+://\w+\.\w+\.\w+/?[\w\.\?=#]*", text)
    for item in emails + urls:
      item_norm = item
      for symbol in symbols:
        item_norm = item_norm.replace(symbol, f' {self.symbols2lang2word[symbol][lang]} ')
      text = text.replace(item, item_norm)
    return text
