# coding: utf-8
# Symbols
PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?'
SPACE = ' '
_SILENCES = ['sp', 'spn', 'sil']

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = list(PAD + EOS + VALID_CHARS) + _SILENCES

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}

# Dictionaries
ko_dict = {
        "quote_checker": """([`"'＂“‘’”’])(.+?)([`"'＂“‘’”’])""",
        "number_checker": "([+-]?\d{1,3},\d{3}(?!\d)|[+-]?\d+)[\.]?\d*", # "([+-]?\d[\d,]*)[\.]?\d*"
        "count_checker": "(시|명|가지|살|마리|포기|송이|수|톨|통|점|개(?!월)|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)",
        
        "num_to_kor": {
                '0': '영',
                '1': '일',
                '2': '이',
                '3': '삼',
                '4': '사',
                '5': '오',
                '6': '육',
                '7': '칠',
                '8': '팔',
                '9': '구',
        },

        "num_to_kor1": [""] + list("일이삼사오육칠팔구"),
        "num_to_kor2": [""] + list("만억조경해"),
        "num_to_kor3": [""] + list("십백천"),
        "count_to_kor1": [""] + ["한","두","세","네","다섯","여섯","일곱","여덟","아홉"], # [""] + ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
        
        "count_tenth_dict": {
                "십": "열",
                "두십": "스물",
                "세십": "서른",
                "네십": "마흔",
                "다섯십": "쉰",
                "여섯십": "예순",
                "일곱십": "일흔",
                "여덟십": "여든",
                "아홉십": "아흔",
        },
        
        "unit_to_kor": {
                '%': '퍼센트',
                'ml': '밀리리터',
                'cm': '센치미터',
                'mm': '밀리미터',
                'km': '킬로미터',
                'kg': '킬로그람',
                'm': '미터',
        },
        
        "upper_to_kor": {
                'A': '에이',
                'B': '비',
                'C': '씨',
                'D': '디',
                'E': '이',
                'F': '에프',
                'G': '지',
                'H': '에이치',
                'I': '아이',
                'J': '제이',
                'K': '케이',
                'L': '엘',
                'M': '엠',
                'N': '엔',
                'O': '오',
                'P': '피',
                'Q': '큐',
                'R': '알',
                'S': '에스',
                'T': '티',
                'U': '유',
                'V': '브이',
                'W': '더블유',
                'X': '엑스',
                'Y': '와이',
                'Z': '지',
        },
        
        "english_dictionary": {
                'TV': '티비',
                'CCTV': '씨씨티비',
                'cctv': '씨씨티비',
                'cc': '씨씨',
                'Apple': '애플',
                'lte': '엘티이',
                'KG': '킬로그람',
                'x': '엑스',
                'z': '제트',
                'Yo': '요',
                'YOLO': '욜로',
                'Gone': '건',
                'gone': '건',
                'Have': '헤브',
                'p': '피',
                'ppt': '피피티',
                'suv': '에스유브이',
        },
        
        "etc_dictionary": {
                '1+1': '원 플러스 원',
                '+': '플러스',
                'MP3': '엠피쓰리',
                '5G': '파이브지',
                '4G': '포지',
                '3G': '쓰리지',
                '2G': '투지',
                'A/S': '에이 에스',
                '1/3':'삼분의 일',
                'greentea907': '그린티 구공칠',
                'CNT 123': '씨엔티 일이삼',
                '14학번': '일사 학번',
                '7011번': '칠공일일번',
                'P8학원': '피에잇 학원',
                '102마리': '백 두마리',
                '20명': '스무명',
        }
}

