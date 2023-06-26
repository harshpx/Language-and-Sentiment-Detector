# necessary libraries
from flask import Flask,render_template,request
import pickle
import re
import numpy
import contractions
from spellchecker import SpellChecker
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# loading models and encoders
language_model = pickle.load(open('language_model.pkl','rb'))
language_enc = pickle.load(open('language_encoder.pkl','rb'))

emotion_model = pickle.load(open('emotion_model.pkl','rb'))
emotion_enc = pickle.load(open('emotion_encoder.pkl','rb'))

sentiment_model = pickle.load(open('sentiment_model.pkl','rb'))
sentiment_enc = pickle.load(open('sentiment_encoder.pkl','rb'))

# function for NLP
def nlp(text):
    def remove_emoji(text):
        emoji_pattern = re.compile(
          '['
          u'\U0001F600-\U0001F64F'  # emoticons
          u'\U0001F300-\U0001F5FF'  # symbols & pictographs
          u'\U0001F680-\U0001F6FF'  # transport & map symbols
          u'\U0001F1E0-\U0001F1FF'  # flags
          u'\U00002702-\U000027B0'
          u'\U000024C2-\U0001F251'
          ']+',
          flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def emoticons_to_text(text):
        EMOTICONS = {
            u"xD":"Funny face", u"XD":"Funny face",
            u":3":"Happy face", u":o":"Happy face",
            u"=D":"Laughing",
            u"D:":"Sadness", u"D;":"Great dismay", u"D=":"Great dismay",
            u":O":"Surprise", u":‑O":"Surprise", u":‑o":"Surprise", u":o":"Surprise", u"o_O":"Surprise",
            u":-0":"Shock", u":X":"Kiss", u";D":"Wink or smirk",
            u":p":"cheeky, playful", u":b":"cheeky, playful", u"d:":"cheeky, playful",
            u"=p":"cheeky, playful", u"=P":"cheeky, playful",
            u":L":"annoyed", u":S":"annoyed", u":@":"annoyed",
            u":$":"blushing", u":x":"Sealed lips",
            u"^.^":"Laugh", u"^_^":"Laugh",
            u"T_T":"Sad", u";_;":"Sad", u";n;":"Sad", u";;":"Sad", u"QQ":"Sad"
        }
        for emot in EMOTICONS:
            text = re.sub(emot, EMOTICONS[emot], text)
        return text
    
    def abbr_to_text(text):
        sample_abbr = {"$" : " dollar ", "€" : " euro ", "4ao" : "for adults only", "a.m" : "before midday", "a3" : "anytime anywhere anyplace",
                        "aamof" : "as a matter of fact", "acct" : "account", "adih" : "another day in hell", "afaic" : "as far as i am concerned",
                        "afaict" : "as far as i can tell", "afaik" : "as far as i know", "afair" : "as far as i remember", "afk" : "away from keyboard",
                        "app" : "application", "approx" : "approximately", "apps" : "applications", "asap" : "as soon as possible", "asl" : "age, sex, location",
                        "atk" : "at the keyboard", "ave." : "avenue", "aymm" : "are you my mother", "ayor" : "at your own risk",
                        "b&b" : "bed and breakfast", "b+b" : "bed and breakfast", "b.c" : "before christ", "b2b" : "business to business",
                        "b2c" : "business to customer", "b4" : "before", "b4n" : "bye for now", "b@u" : "back at you", "bae" : "before anyone else",
                        "bak" : "back at keyboard", "bbbg" : "bye bye be good", "bbc" : "british broadcasting corporation", "bbias" : "be back in a second",
                        "bbl" : "be back later", "bbs" : "be back soon", "be4" : "before", "bfn" : "bye for now", "blvd" : "boulevard", "bout" : "about",
                        "brb" : "be right back", "bros" : "brothers", "brt" : "be right there", "bsaaw" : "big smile and a wink", "btw" : "by the way",
                        "bwl" : "bursting with laughter", "c/o" : "care of", "cet" : "central european time", "cf" : "compare", "cia" : "central intelligence agency",
                        "csl" : "can not stop laughing", "cu" : "see you", "cul8r" : "see you later", "cv" : "curriculum vitae", "cwot" : "complete waste of time",
                        "cya" : "see you", "cyt" : "see you tomorrow", "dae" : "does anyone else", "dbmib" : "do not bother me i am busy", "diy" : "do it yourself",
                        "dm" : "direct message", "dwh" : "during work hours", "e123" : "easy as one two three", "eet" : "eastern european time", "eg" : "example",
                        "embm" : "early morning business meeting", "encl" : "enclosed", "encl." : "enclosed", "etc" : "and so on", "faq" : "frequently asked questions",
                        "fawc" : "for anyone who cares", "fb" : "facebook", "fc" : "fingers crossed", "fig" : "figure","fimh" : "forever in my heart", 
                        "ft." : "feet", "ft" : "featuring", "ftl" : "for the loss", "ftw" : "for the win", "fwiw" : "for what it is worth", "fyi" : "for your information",
                        "g9" : "genius", "gahoy" : "get a hold of yourself", "gal" : "get a life", "gcse" : "general certificate of secondary education",
                        "gfn" : "gone for now", "gg" : "good game", "gl" : "good luck", "glhf" : "good luck have fun", "gmt" : "greenwich mean time",
                        "gmta" : "great minds think alike", "gn" : "good night", "g.o.a.t" : "greatest of all time", "goat" : "greatest of all time",
                        "goi" : "get over it", "gps" : "global positioning system", "gr8" : "great", "gratz" : "congratulations", "gyal" : "girl",
                        "h&c" : "hot and cold", "hp" : "horsepower", "hr" : "hour", "hrh" : "his royal highness", "ht" : "height", "ibrb" : "i will be right back",
                        "ic" : "i see", "icq" : "i seek you", "icymi" : "in case you missed it","idc" : "i do not care", "idgadf" : "i do not give a damn fuck",
                        "idgaf" : "i do not give a fuck", "idk" : "i do not know", "ie" : "that is", "i.e" : "that is", "iykyk":"if you know you know",
                        "ifyp" : "i feel your pain", "IG" : "instagram", "ig":"instagram", "iirc" : "if i remember correctly", "ilu" : "i love you",
                        "ily" : "i love you", "imho" : "in my humble opinion", "imo" : "in my opinion", "imu" : "i miss you", "iow" : "in other words",
                        "irl" : "in real life", "j4f" : "just for fun", "jic" : "just in case", "jk" : "just kidding", "jsyk" : "just so you know",
                        "l8r" : "later", "lb" : "pound", "lbs" : "pounds", "ldr" : "long distance relationship", "lmao" : "laugh my ass off", "luv":"love",
                        "lmfao" : "laugh my fucking ass off", "lol" : "laughing out loud", "ltd" : "limited","ltns" : "long time no see", "m8" : "mate",
                        "mf" : "motherfucker", "mfs" : "motherfuckers", "mfw" : "my face when","mofo" : "motherfucker","mph" : "miles per hour","mr" : "mister",
                        "mrw" : "my reaction when", "ms" : "miss", "mte" : "my thoughts exactly", "nagi" : "not a good idea", "nbc" : "national broadcasting company",
                        "nbd" : "not big deal", "nfs" : "not for sale", "ngl" : "not going to lie", "nhs" : "national health service", "nrn" : "no reply necessary",
                        "nsfl" : "not safe for life", "nsfw" : "not safe for work", "nth" : "nice to have", "nvr" : "never", "nyc" : "new york city",
                        "oc" : "original content", "og" : "original", "ohp" : "overhead projector", "oic" : "oh i see", "omdb" : "over my dead body",
                        "omg" : "oh my god", "omw" : "on my way", "p.a" : "per annum", "p.m" : "after midday", "pm" : "prime minister", "poc" : "people of color",
                        "pov" : "point of view", "pp" : "pages", "ppl" : "people", "prw" : "parents are watching", "ps" : "postscript", "pt" : "point",
                        "ptb" : "please text back", "pto" : "please turn over","qpsa" : "what happens", "ratchet" : "rude", "rbtl" : "read between the lines",
                        "rlrt" : "real life retweet",  "rofl" : "rolling on the floor laughing", "roflol" : "rolling on the floor laughing out loud",
                        "rotflmao" : "rolling on the floor laughing my ass off", "rt" : "retweet", "ruok" : "are you ok", "sfw" : "safe for work", "sk8" : "skate",
                        "smh" : "shake my head", "sq" : "square", "srsly" : "seriously", 
                        "ssdd" : "same stuff different day", "tbh" : "to be honest", "tbs" : "tablespooful", "tbsp" : "tablespooful", "tfw" : "that feeling when",
                        "thks" : "thank you", "tho" : "though", "thx" : "thank you", "tia" : "thanks in advance", "til" : "today i learned", "tl;dr" : "too long i did not read", "tldr" : "too long i did not read",
                        "tmb" : "tweet me back", "tntl" : "trying not to laugh", "ttyl" : "talk to you later", "u" : "you", "u2" : "you too", "u4e" : "yours for ever",
                        "utc" : "coordinated universal time", "w/" : "with", "w/o" : "without", "w8" : "wait", "wassup" : "what is up", "wb" : "welcome back",
                        "wtf" : "what the fuck", "wtg" : "way to go", "wtpa" : "where the party at", "wuf" : "where are you from", "wuzup" : "what is up",
                        "wywh" : "wish you were here", "yd" : "yard", "ygtr" : "you got that right", "ynk" : "you never know", "zzz" : "sleeping bored and tired"
                    }
        sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')
        text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text)
        return text
    
    def correct_spellings(text):
        spell = SpellChecker()
        corrected_text = []
        misspelled_words = spell.unknown(text.split())
        for word in text.split():
            if word in misspelled_words:
                corrected_text.append(spell.correction(word))
            else:
                corrected_text.append(word)
        return " ".join(str(word) for word in corrected_text)
    
    
    def lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        words = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stopwords.words('english')])
        return words
    
    text = remove_emoji(text)
    text = emoticons_to_text(text)
    text = str(text).lower()
    text = abbr_to_text(text)
    text = re.sub(r'<.*?>','', text)  # HTML tags
    text = re.sub(r'https?://\S+|www\.\S+','',text)  # URLs
    text = re.sub(r'@\S+','',text)  # Mentions
    text = re.sub(r'&\S+','',text)  # html characters
    text = re.sub(r'[^\x00-\x7f]','',text)  # non-ASCII
    text = contractions.fix(text)  # update contractions
    text = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)  # punctuations, special chars
    text = re.sub(r'\s+', ' ', text)
    text = correct_spellings(text)
    text = lemmatize(text)
    return text

def clean_txt(text):
        text=text.lower()
        text=re.sub(r'[^\w\s]',' ',text)
        text=re.sub(r'[_0-9]',' ',text)
        text=re.sub(r'\s\s+',' ',text)
        return text


# define app routs
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/get_started', methods=['POST'])
def start():
    return render_template('selector.html')

@app.route('/language', methods=['POST'])
def language():
    return render_template('language_input.html')

@app.route('/emotion', methods=['POST'])
def emotion():
    return render_template('emotion_input.html')


@app.route('/predict_language', methods=['POST'])
def detect_language():
    text = str(request.form.get('textInput'))
    pred = language_model.predict([clean_txt(text)])
    res = language_enc.inverse_transform(pred)
    output = f'Predicted Language: {str(res[0])}'
    return render_template('language_output.html',result=output)

@app.route('/back_language', methods=['POST'])
def back_lang():
    return render_template('language_input.html')


@app.route('/predict_emotion', methods=['POST'])
def detect_emotion():
    text = str(request.form.get('textInput'))

    pred1 = emotion_model.predict([nlp(text)])
    res1 = emotion_enc.inverse_transform(pred1)[0]

    pred2 = sentiment_model.predict([nlp(text)])
    res2 = sentiment_enc.inverse_transform(pred2)[0]

    output1 = f"Predicted Emotion: {str(res1)}"
    output2 = f"Predicted Sentiment: {str(res2)}"
    output = output1+"\n"+output2
    return render_template('emotion_output.html',result=output)

@app.route('/back_emotion', methods=['POST'])
def back_emot():
    return render_template('emotion_input.html')

# run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)