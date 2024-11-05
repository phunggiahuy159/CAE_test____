import pandas as pd
from pyvi import ViTokenizer
import numpy as np 
import string 
import re 


aspect2idx={'CAMERA' : 0,
            'FEATURES' : 1,
            'BATTERY':2,
            'PERFORMANCE' : 3,
            'DESIGN' : 4,
            'GENERAL' : 5,
            'PRICE' : 6,
            'SCREEN' : 7,
            'SER&ACC' : 8,
            'STORAGE' : 9 
}   
sentiment2idx={'Positive':2,
               'Neutral':1,
               'Negative':0,
}
sample='{FEATURES#Negative};{PERFORMANCE#Positive};{SER&ACC#Positive}'
num_aspect = 10
num_sentiment = 3

def convert_label(text):
    text = text.replace('{OTHERS};', '')
    all_aspect = text.split(';')
    all_aspect = [x.strip(r"{}") for x in all_aspect if x]  
    # res=[-1 for x in range(2*num_aspect)]
    # res=np.array(res)
    res = np.zeros(2 * num_aspect)
    for x in all_aspect:
        cate, sent = x.split('#')
        if cate in aspect2idx and sent in sentiment2idx:
            cate_value = aspect2idx[cate]
            sent_value = sentiment2idx[sent]
            res[cate_value] = 1
            res[cate_value + num_aspect] = sent_value
    for idx in range(num_aspect):
        if res[idx]==0:
            res[idx+num_aspect]=-1

    
    return res
print(convert_label(sample))
    

punc = string.punctuation
tokenizer = ViTokenizer.tokenize

train_df = pd.read_csv("D:\code\intro_ai_ABSA\CAE____\Train.csv")
# dev_df = pd.read_csv("Dev.csv")
test_df = pd.read_csv("D:\code\intro_ai_ABSA\CAE____\Test.csv")

def lowercase(df):
    df['comment'] = df['comment'].str.lower()
    return df

def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))

def final_rmv_punc(df):
    df['comment'] = df['comment'].apply(remove_punc)
    return df

def remove_num(df):
    df['comment'] = df['comment'].replace(to_replace=r'\d', value='', regex=True)
    return df

def tokenize(df):
    df['comment'] = df['comment'].apply(tokenizer)
    return df
def remove_emote(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"  # other symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U0001f926-\U0001f937"  # gestures
        u"\U0001F1F2"             # specific characters
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"          # gender symbols
        "]+", flags=re.UNICODE)
    
    # Substitute emojis with a space
    text = emoji_pattern.sub(r" ", text)
    return text
def final_remove_emote(df):
    df['comment'] = df['comment'].apply(remove_emote)    
    return df
def remove_newline(text):
    # Replace newline characters with a single space
    return text.replace('\n', ' ')
def final_remove_newline(df):
    df['comment'] = df['comment'].apply(remove_newline)
    return df

def normalize_acronyms(sent):
    text = sent
    replace_list = {
        'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ', ' okie' : ' ok',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' tích cực ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' tích cực ','hehe': ' tích cực ','hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',
        ' lol ': ' tiêu cực ',' cc ': ' tiêu cực ','cute': u' dễ thương ','huhu': ' tiêu cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích cực ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tệ','fresh': ' tươi ','sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích cực ',
        ' por ': u' tệ ',' poor ': u' tệ ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
        ' h ' : u' giờ', 
        ' e ' : u' em'
        
    }
    for k, v in replace_list.items():
        text = text.replace(k, v)
    return text
def final_normalize_acronyms(df):
    df['comment'] = df ['comment'].apply(normalize_acronyms)
    return df   

def preprocess(df):
    df.drop(['n_star', 'date_time'],axis=1, inplace = True)
    df = final_remove_emote(df)
    df = final_remove_newline(df)
    df = remove_num(df)
    df = lowercase(df)
    df = final_normalize_acronyms(df)
    df = final_rmv_punc(df)
    df = tokenize(df)

    df['label'] = df['label'].apply(convert_label)
    return df

train = preprocess(train_df)
test = preprocess(test_df)
test.to_csv('Test_final.csv')
# train.to_csv('Train_preprocessed_with_-1.csv')
