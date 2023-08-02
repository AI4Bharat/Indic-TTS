#! /usr/bin/env python
# encoding: utf-8

import re

non_chars_regex = re.compile(r'[^\w]')

class ParagraphHandler():

    def __init__(self, max_text_len=512):
        self.L = max_text_len
    
    def split_text(self, text:str, delimiter='.'):
        '''Splits text at delimiter into paragraphs of max. length self.L'''
        delimiter = ' ' if delimiter not in text else delimiter
        if delimiter not in text:
            return [text]
        
        paragraphs = []
        l_pos, r_pos = 0, 0
        while r_pos < len(text):
            r_pos = l_pos + self.L
            if r_pos >= len(text): # append last paragraph. 
                paragraphs.append(text[l_pos:len(text)])
                break
            while delimiter is not None and text[r_pos] != delimiter and r_pos > l_pos and r_pos > 0: # find nearest delimiter < r_pos to split paragraph at.
                r_pos -= 1
            extracted_paragraph = text[l_pos:r_pos+1]
            extracted_paragraph_without_special_chars = non_chars_regex.sub('', extracted_paragraph)
            if extracted_paragraph_without_special_chars:
                paragraphs.append(extracted_paragraph)
            l_pos = r_pos + 1  # handle next paragraph
        return paragraphs


if __name__ == '__main__':
    text = "The following are quotes from A.P.J. Abdul Kalam. To succeed in your mission, you must have single-minded devotion to your goal. Look at the sky. We are not alone. The whole universe is friendly to us and conspires only to give the best to those who dream and work. The youth need to be enabled to become job generators from job seekers. If four things are followed - having a great aim, acquiring knowledge, hard work, and perseverance - then anything can be achieved. Where there is righteousness in the heart, there is beauty in the character. When there is beauty in the character, there is harmony in the home. When there is harmony in the home, there is order in the nation. When there is order in the nation, there is peace in the world. Great teachers emanate out of knowledge, passion and compassion. Let me define a leader. He must have vision and passion and not be afraid of any problem. Instead, he should know how to defeat it. Most importantly, he must work with integrity."
    print('LENGTH: ', len(text))  # 988

    paragraph_handler = ParagraphHandler()
    paragraphs = paragraph_handler.split_text(text)
    for p in paragraphs:
        print(len(p), p)
