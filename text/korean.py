# coding: utf-8
# Code based on 

import re
import os
import ast
import json
from quickspacer import Spacer
from g2pk import G2p
from jamo import hangul_to_jamo, h2j, j2h
from jamo.jamo import _jamo_char_to_hcj
from .korean_dict import JAMO_LEADS, JAMO_VOWELS, JAMO_TAILS, ko_dict

g2p=G2p()
spacer = Spacer(level=3)


def tokenize(text, norm=True):
    """
    Input --- Grapheme in string
    Output --- Phoneme in list

    Example:
        '한글은 위대하다.'  --> ['ᄒ', 'ᅡ', 'ᆫ', 'ᄀ', 'ᅳ', 'ᄅ', 'ᅳ', ' ', 'ᄂ', 'ᅱ', 'ᄃ', 'ᅢ', 'ᄒ', 'ᅡ', 'ᄃ', 'ᅡ', '.']
    """
    if norm:
        text = normalize(text)
    text = g2p(text)
    tokens = list(hangul_to_jamo(text))

    return tokens


def detokenize(tokens):
    """s
    Input --- Grapheme or Phoneme in list
    Output --- Grapheme or Phoneme in string

    Example:
        ['ᄒ', 'ᅡ', 'ᆫ', 'ᄀ', 'ᅳ', 'ᆯ', 'ᄋ', 'ᅳ', 'ᆫ', ' ', 'ᄋ', 'ᅱ', 'ᄃ', 'ᅢ', 'ᄒ', 'ᅡ', 'ᄃ', 'ᅡ', '.'] --> '한글은 위대하다.'
        ['ᄒ', 'ᅡ', 'ᆫ', 'ᄀ', 'ᅳ', 'ᄅ', 'ᅳ', ' ', 'ᄂ', 'ᅱ', 'ᄃ', 'ᅢ', 'ᄒ', 'ᅡ', 'ᄃ', 'ᅡ', '.'] --> '한그르 뉘대하다.'
    """
    tokens = h2j(tokens)

    idx = 0
    text = ""
    candidates = []

    while True:
        if idx >= len(tokens):
            text += _get_text_from_candidates(candidates)
            break

        char = tokens[idx]
        mode = _get_mode(char)

        if mode == 0:
            text += _get_text_from_candidates(candidates)
            candidates = [char]
        elif mode == -1:
            text += _get_text_from_candidates(candidates)
            text += char
            candidates = []
        else:
            candidates.append(char)

        idx += 1
    return text


def _get_mode(char):
    if char in JAMO_LEADS:
        return 0
    elif char in JAMO_VOWELS:
        return 1
    elif char in JAMO_TAILS:
        return 2
    else:
        return -1


def _get_text_from_candidates(candidates):
    if len(candidates) == 0:
        return ""
    elif len(candidates) == 1:
        return _jamo_char_to_hcj(candidates[0])
    else:
        return j2h(**dict(zip(["lead", "vowel", "tail"], candidates)))


def compare_sentence_with_jamo(text1, text2):
    return h2j(text1) != h2j(text2)


def normalize(text):
    """
    Transliterate input text into Hangul grapheme.
    """
    text = text.strip()
    text = re.sub('\(\d+일\)', '', text)
    text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)

    text = normalize_with_dictionary(text, ko_dict["etc_dictionary"])
    text = normalize_english(text)
    text = re.sub('[a-zA-Z]+', normalize_upper, text)

    text = normalize_quote(text)
    text = normalize_number(text)
    text = normalize_nonchar(text)
    text = spacer.space([text])[0]

    return text


def normalize_nonchar(text, inference=False):
    return re.sub(r"\{[^\w\s]?\}", "{sp}", text) if inference else\
            re.sub(r"[^\w\s]?", "", text)


def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in ko_dict["english_dictionary"]:
            return ko_dict["english_dictionary"].get(word)
        else:
            return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


def normalize_upper(text):
    text = text.group(0)

    if all([char.isupper() for char in text]):
        return "".join(ko_dict["upper_to_kor"][char] for char in text)
    else:
        return text


def normalize_quote(text):
    def fn(found_text):
        from nltk import sent_tokenize # NLTK doesn't along with multiprocessing

        found_text = found_text.group()
        unquoted_text = found_text[1:-1]

        sentences = sent_tokenize(unquoted_text)
        return " ".join(["'{}'".format(sent) for sent in sentences])

    return re.sub(ko_dict["quote_checker"], fn, text)


def normalize_number(text):
    text = normalize_with_dictionary(text, ko_dict["unit_to_kor"])
    text = re.sub(ko_dict["number_checker"] + ko_dict["count_checker"],
            lambda x: number_to_korean(x, True), text)
    text = re.sub(ko_dict["number_checker"],
            lambda x: number_to_korean(x, False), text)
    return text


def number_to_korean(num_str, is_count=False):
    zero_cnt = 0
    if is_count:
        num_str, unit_str = num_str.group(1), num_str.group(2)
    else:
        num_str, unit_str = num_str.group(), ""

    num_str = num_str.replace(',', '')
    
    if is_count and len(num_str) > 2:
        is_count = False

    if len(num_str) > 1 and num_str.startswith("0") and '.' not in num_str:
        for n in num_str:
            zero_cnt += 1 if n == "0" else 0
        num_str = num_str[zero_cnt:]
    
    kor = ""
    if num_str != '':
        num = ast.literal_eval(num_str)

        if num == 0:
            return "영" + (unit_str if unit_str else "")

        check_float = num_str.split('.')
        if len(check_float) == 2:
            digit_str, float_str = check_float
        elif len(check_float) >= 3:
            raise Exception(" [!] Wrong number format")
        else:
            digit_str, float_str = check_float[0], None

        if is_count and float_str is not None:
            raise Exception(" [!] `is_count` and float number does not fit each other")

        digit = int(digit_str)

        if digit_str.startswith("-") or digit_str.startswith("+"):
            digit, digit_str = abs(digit), str(abs(digit))

        size = len(str(digit))
        tmp = []

        for i, v in enumerate(digit_str, start=1):
            v = int(v)

            if v != 0:
                if is_count:
                    tmp += ko_dict["count_to_kor1"][v]
                else:
                    tmp += ko_dict["num_to_kor1"][v]
                    if v == 1 and i != 1 and i != len(digit_str):
                        tmp = tmp[:-1]
                tmp += ko_dict["num_to_kor3"][(size - i) % 4]

            if (size - i) % 4 == 0 and len(tmp) != 0:
                kor += "".join(tmp)
                tmp = []
                kor += ko_dict["num_to_kor2"][int((size - i) / 4)]

        if is_count:
            if kor.startswith("한") and len(kor) > 1:
                kor = kor[1:]

            if any(word in kor for word in ko_dict["count_tenth_dict"]):
                kor = re.sub(
                        '|'.join(ko_dict["count_tenth_dict"].keys()),
                        lambda x: ko_dict["count_tenth_dict"][x.group()], kor)

        if not is_count and kor.startswith("일") and len(kor) > 1:
            kor = kor[1:]

        if float_str is not None and float_str != "":
            kor += "영" if kor == "" else ""
            kor += "쩜 "
            kor += re.sub('\d', lambda x: ko_dict["num_to_kor"][x.group()], float_str)
    
    if num_str.startswith("+"):
        kor = "플러스 " + kor
    elif num_str.startswith("-"):
        kor = "마이너스 " + kor
    if zero_cnt > 0:
        kor = "공"*zero_cnt + kor

    return kor + unit_str


def test_normalize(texts):
    for text in texts:
        raw = text
        norm = normalize(text)

        print("="*30)
        print(raw)
        print(norm)


if __name__ == "__main__":
    test_inputs = [
        "JTBC는 JTBCs를 DY는 A가 Absolute",
        "오늘(13일) 3,600마리 강아지가",
        "60.3%",
        '"저돌"(猪突) 입니다.',
        '비대위원장이 지난 1월 이런 말을 했습니다. “난 그냥 산돼지처럼 돌파하는 스타일이다”',
        "지금은 -12.35%였고 종류는 5가지와 19가지, 그리고 55가지였다",
        "JTBC는 TH와 K 양이 2017년 9월 12일 오후 12시에 24살이 된다",
        "이렇게 세트로 98,000원인데, 지금 세일 중이어서, 78,400원이에요.",
        "이렇게 세트로 98000원인데, 지금 세일 중이어서, 78400원이에요.",
        "저, 토익 970점이요.",
        "원래대로라면은 0점 처리해야 하는데.",
        "진짜? 그럼 너한테 한 두 마리만 줘도 돼?",
        "내가 화분이 좀 많아서. 그래도 17평에서 20평은 됐으면 좋겠어. 요즘 애들, 많이 사는 원룸, 그런데는 말고.",
        "매매는 3억까지. 전세는 1억 5천. 그 이상은 안돼.",
        "1억 3천이요.",
        "지금 3개월입니다.",
        "기계값 200만원 짜리를. 30개월 할부로 300만원에 파셨잖아요!",
        "오늘(13일) 99통 강아지가",
        "이제 55개.. 째예요.",
        "이제 55개월.. 째예요.",
        "한 근에 3만 5천 원이나 하는 1++ 등급 한우라니까!",
        "한 근에 3만 5천 원이나 하는 A+ 등급 한우라니까!",
        "19,22,30,34,39,44+36",
        "그거 1+1으로 프로모션 때려버리자.",
        "아 테이프는 너무 우리 때 얘기인가? 그 MP3 파일 같은 거 있잖아. 영어 중국어 이런 거. 영어 책 읽어주고 그런 거.",
        "231 cm야.",
        "1 cm야.",
        "21 cm야.",
        "110 cm야.",
        "21마리야.",
        "아, 시력은 알고 있어요. 왼쪽 0.3이고 오른쪽 0.1이요.",
        "왼쪽 0점",
        "우리 스마트폰 쓰기 전에 공일일 번호였을때. 그 때 썼던 전화기를 2G라고 하거든?",
        "102마리 강아지.",
        "87. 105. 120. 네. 100 넘었어요!",
    ]

    test_normalize(test_inputs)