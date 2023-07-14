class GoogleTranslator:
  def __init__(self):
    from translators.server import google, _google
    self._translate = google

    google("Testing...")
    self.supported_languages = set(_google.language_map['en'])
    self.custom_lang_map = {
        "mni": "mni-Mtei",
        "raj": "hi",
    }

  def translate(self, text, from_lang, to_lang):
    if from_lang in self.custom_lang_map:
      from_lang = self.custom_lang_map[from_lang]
    elif from_lang not in self.supported_languages:
      return text
    
    if to_lang in self.custom_lang_map:
      to_lang = self.custom_lang_map[to_lang]
    elif to_lang not in self.supported_languages:
      return text
    
    return self._translate(text, from_language=from_lang, to_language=to_lang)
  
  def __call__(self, **kwargs):
    return self.translate(**kwargs)
