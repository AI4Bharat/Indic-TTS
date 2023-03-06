from ai4bharat.transliteration import XlitEngine
xlit_engine = XlitEngine()

INPUT_WORD = "namaste"
print("Input word:", INPUT_WORD)

for lang_code in xlit_engine.all_supported_langs:
    res = xlit_engine.translit_word(INPUT_WORD, lang_code)
    print(res)
