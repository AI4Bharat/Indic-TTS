from ai4bharat.transliteration import XlitEngine

INPUT_WORD = "namaste"
print("Input word:", INPUT_WORD)

# xlit_engine = XlitEngine()
# for lang_code in xlit_engine.all_supported_langs:
#     res = xlit_engine.translit_word(INPUT_WORD, lang_code)
#     print(res)

xlit_engine = XlitEngine("hi")
res = xlit_engine.translit_word(INPUT_WORD)
print("Hindi output:", res)
