""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''


"""
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

print(SPACE_ID)
"""

### add Japanese phonomes by pyopenjtalk_g2p_prosody ###
symbols = [
    '_' , '#' , '$' , '[' , '?' , ']' , '^' ,  
    'a' , 'b' , 'by', 'ch', 'cl', 'd' , 'dy',
    'e' , 'f' , 'g' , 'gy', 'h' , 'hy', 'i' ,
    'j' , 'k' , 'ky', 'm' , 'my', 'n' , 'N' , 
    'ny', 'o' , 'p' , 'py', 'r' , 'ry', 's' , 
    'sh', 't' , 'ts', 'ty', 'u' , 'v'  , 'w', 
    'y' , 'z' ]
########################################################