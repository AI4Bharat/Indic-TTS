#! /usr/bin/env python
# encoding: utf-8

import re

from indicnlp.tokenize.sentence_tokenize import sentence_split

non_chars_regex = re.compile(r"[^\w]")


class ParagraphHandler:
    def __init__(self, max_text_len=512):
        self.L = max_text_len

    def split_text_forced(self, text: str):
        sents = []
        cur_sent = ""
        for word in text.split(" "):
            if len(cur_sent + word) > self.L:
                sents.append(cur_sent.strip())
                cur_sent = ""

            cur_sent += word + " "

        sents.append(cur_sent)
        return sents

    def split_text(self, text: str, split_lang: str):
        """Splits text at delimiter into paragraphs of max. length self.L"""
        paragraphs = sentence_split(text, lang=split_lang)

        further_split_paras = []
        for paragraph in paragraphs:
            if len(paragraph) > self.L:
                further_split_paras.extend(self.split_text_forced(paragraph))
            else:
                if len(paragraph) < 10 and len(further_split_paras) > 0:
                    further_split_paras[-1] += paragraph
                else:
                    further_split_paras.append(paragraph)

        return further_split_paras


if __name__ == "__main__":
    text = "The following are quotes from A.P.J. Abdul Kalam. To succeed in your mission, you must have single-minded devotion to your goal. Look at the sky. We are not alone. The whole universe is friendly to us and conspires only to give the best to those who dream and work. The youth need to be enabled to become job generators from job seekers. If four things are followed - having a great aim, acquiring knowledge, hard work, and perseverance - then anything can be achieved. Where there is righteousness in the heart, there is beauty in the character. When there is beauty in the character, there is harmony in the home. When there is harmony in the home, there is order in the nation. When there is order in the nation, there is peace in the world. Great teachers emanate out of knowledge, passion and compassion. Let me define a leader. He must have vision and passion and not be afraid of any problem. Instead, he should know how to defeat it. Most importantly, he must work with integrity."
    print("LENGTH: ", len(text))  # 988

    paragraph_handler = ParagraphHandler()
    paragraphs = paragraph_handler.split_text(text)
    for p in paragraphs:
        print(len(p), p)
