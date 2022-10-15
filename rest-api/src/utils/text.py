import re

num_str_regex = re.compile("\d{1,3}(?:(?:,\d{2,3})+|(?:\d+))?(?:\.\d+)?")
def get_all_numbers_from_string(text):
  return num_str_regex.findall(text)

from num2words import num2words
import traceback
from .translator import GoogleTranslator

class TextNormalizer:
  def __init__(self):
    self.translator = GoogleTranslator()

  def convert_numbers_to_words(self, text, lang):
    num_strs = get_all_numbers_from_string(text)
    numbers = [float(num_str.replace(',', '')) for num_str in num_strs]

    # TODO: Use this library once stable: https://github.com/sutariyaraj/indic-num2words
    num_words = [num2words(num, lang="en_IN") for num in numbers]
    if lang != "en":
      try:
        translated_num_words = [self.translator(text=num_word, from_lang="en", to_lang=lang) for num_word in num_words]
        num_words = translated_num_words
      except:
        traceback.print_exc()
    
    for num_str, num_word in zip(num_strs, num_words):
      text = text.replace(num_str, ' '+num_word+' ', 1)
    return text.replace("  ", ' ')
