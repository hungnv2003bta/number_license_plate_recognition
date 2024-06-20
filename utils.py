import easyocr
import string

reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'Z': '2',
                    'I': '1',
                    'J': '3',
                    'B': '3',
                    'A': '4',
                    'L': '4',
                    'G': '6',
                    'S': '5',
                    'U': '0',
                    'D': '0',
                    'T': '1',
                    'Q': '0',
                    }
dict_int_to_char = {'0': 'D',
                    '2': 'Z',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8': 'B'}

def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    len_text = len(text)
    if (len_text == 8):
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
            (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
            (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
            (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
            
            return True
        else:
            print('License plate text does not comply with the format.')
            return False
    
    elif (len_text == 7):
        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
            (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
            (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()):
            return True
        else:
            print('License plate text does not comply with the format.')
            return False
        

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_char_to_int, 1: dict_char_to_int,3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int,
                2: dict_int_to_char}

    for j in range(len(text)):
        if text[j].isalnum() or text[j] == '.':
            if j in mapping and text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_

def read_license_plate(detections):
    # detections = reader.readtext(license_plate_crop)
    # detections = sorted(detections, key=lambda x: (x[0][1], x[0][0]))
    if detections == []:
        return None, None
    bbox, text, score = detections[0]

    if len(detections) == 2:
        bbox1, text1, score1 = detections[1]
        text = text + text1
        score = (score * 3 + score1 * 5) / 8
    
    text = text.upper()
    # replace all characters not number and alphabet
    text = ''.join(e for e in text if e.isalnum())
    if license_complies_format(text):
        return format_license(text), score
    else:
        return text, "0.0"