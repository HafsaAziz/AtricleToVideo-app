import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import re
import json
import urllib.parse
import os
from gtts import gTTS
import shutil
from PIL import Image, ImageDraw, ImageFont
import textwrap
from io import BytesIO
import glob
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from moviepy.audio.AudioClip import AudioClip
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()
openai.api_key  = os.getenv("OPENAI_API_KEY")

def scrape_text_from_url(url):

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {url}")
    soup = BeautifulSoup(response.content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    text = soup.get_text()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def call_llm_api(article_text, slidenumber , wordnumber, language):

    api_url = "https://a.picoapps.xyz/ask-ai?prompt="
    prompt = '''Article scraped text and content and Data : '''  + article_text + ''' 
    Task :  You are an llm in json mode now, you generate directly a json, no more, no less, where you summarize this article in '''  +  str(slidenumber) + ''' bullet points in a journalist narrative reporting style format that highlight the main key points from the article, and keep the coherence between each bullet point like a story video content, make it in '''  + language + ''' and dont use Unicodes like "u00e9" put a real "é".
    NB:  Make '''  +  str(slidenumber) + ''' short bullet points (arround ''' +  str(wordnumber) + ''' words max per each) in a narrative style like a reporter and all these bullet points summarize and narrate it in a great way that gives the user a great general idea about the article and don't miss the main ideas but try to keep the flow running and coherence between each bullet point in a way where you can read them and feel like you are reading one article. and there is '''  +  str(slidenumber) + ''' bullet point and dont forget that you need to genertare a '''  + language + ''' text.
    Example :  {"summary": ["Bullet point 1", "Bullet point 2", "Bullet point 3",...], "Total": "x" , "Tone": "Tone of the best voice over for it"}
    IMPORTANT : The article text is the only input you have, you can't use any other data or information, you can't use any other source or external data, dont helucinate, dont imagine, dont make up, dont add, dont remove, dont change, dont modify, dont do anything else, just summarize the article in bullet points format that highlight the main key points from the article, no Unicodes, and do it in '''  + language + ''' and Focus on Generating the right characthers and not giving Unicode like in french use é,à,è,ù... please never generate Unicodes.and for numbers dont put a "." like in "1.600" write directly "1600"; and if mentioned use "S.M. ..." for King Mohammed VI. and generate only the json no intro no outro no nothing else, just the json, no more, no less.
    '''
    
    encoded_prompt = urllib.parse.quote(prompt)
    full_url = f"{api_url}{encoded_prompt}"
    response = requests.get(full_url)
    if response.status_code != 200:
        raise Exception(f"Failed to call the LLM API: {response.status_code}")
    result = response.json()

    return result


def save_and_clean_json(response, file_path):
    # First, handle the case where response is a string
    if isinstance(response, str):
        response = json.loads(response.replace('\n', '').replace('\\', ''))
    
    # If response is a dict and contains 'response' key
    if isinstance(response, dict) and 'response' in response:
        response = response['response']
        # If response is still a string, parse it
        if isinstance(response, str):
            response = json.loads(response.replace('\n', '').replace('\\', ''))

    # Write the cleaned JSON to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)
    
    return response


def fix_unicode(text):
     # Preprocess text - replace common Unicode characters
    # French characters
    text = text.replace('\\u00e9', 'é').replace('\\u00e8', 'è').replace('\\u00ea', 'ê')
    text = text.replace('\\u00e0', 'à').replace('\\u00e2', 'â').replace('\\u00f9', 'ù')
    text = text.replace('\\u00fb', 'û').replace('\\u00ee', 'î').replace('\\u00ef', 'ï')
    text = text.replace('\\u00e7', 'ç').replace('\\u0153', 'œ').replace('\\u00e6', 'æ')
    text = text.replace('\\u20ac', '€').replace('\\u00ab', '«').replace('\\u00bb', '»')
    text = text.replace('\\u2013', '–').replace('\\u2014', '—').replace('\\u2018', '‘')
    text = text.replace('\\u2019', '’').replace('\\u201a', '‚').replace('\\u201c', '“')
    text = text.replace('\\u201d', '”').replace('\\u201e', '„').replace('\\u2026', '…')
    text = text.replace('\\u2030', '‰').replace('\\u0152', 'Œ').replace('\\u00a0', ' ')
    text = text.replace('\\u00b0', '°').replace('\\u00a3', '£').replace('\\u00a7', '§')
    text = text.replace('\\u00b7', '·').replace('\\u00bf', '¿').replace('\\u00a9', '©')
    text = text.replace('\\u00ae', '®').replace('\\u2122', '™').replace('\\u00bc', '¼')
    text = text.replace('\\u00bd', '½').replace('\\u00be', '¾').replace('\\u00b1', '±')
    text = text.replace('\\u00d7', '×').replace('\\u00f7', '÷').replace('\\u00a2', '¢')
    text = text.replace('\\u00a5', '¥').replace('\\u00ac', '¬').replace('\\u00b6', '¶')
    text = text.replace('\\u2022', '•')

    # Spanish characters
    text = text.replace('\\u00f1', 'ñ').replace('\\u00ed', 'í').replace('\\u00f3', 'ó')
    text = text.replace('\\u00fa', 'ú').replace('\\u00fc', 'ü').replace('\\u00a1', '¡')
    text = text.replace('\\u00bf', '¿').replace('\\u00e1', 'á').replace('\\u00e9', 'é')
    text = text.replace('\\u00f3', 'ó').replace('\\u00fa', 'ú').replace('\\u00fc', 'ü')
    # German characters
    text = text.replace('\\u00df', 'ß').replace('\\u00e4', 'ä').replace('\\u00f6', 'ö')
    text = text.replace('\\u00fc', 'ü')

    # Italian characters
    text = text.replace('\\u00e0', 'à').replace('\\u00e8', 'è').replace('\\u00e9', 'é')
    text = text.replace('\\u00ec', 'ì').replace('\\u00f2', 'ò').replace('\\u00f9', 'ù')
    text = text.replace('\\u00f9', 'ù')

    # Russian characters
    text = text.replace('\\u0410', 'А').replace('\\u0411', 'Б').replace('\\u0412', 'В')
    text = text.replace('\\u0413', 'Г').replace('\\u0414', 'Д').replace('\\u0415', 'Е')
    text = text.replace('\\u0416', 'Ж').replace('\\u0417', 'З').replace('\\u0418', 'И')
    text = text.replace('\\u0419', 'Й').replace('\\u041a', 'К').replace('\\u041b', 'Л')
    text = text.replace('\\u041c', 'М').replace('\\u041d', 'Н').replace('\\u041e', 'О')
    text = text.replace('\\u041f', 'П').replace('\\u0420', 'Р').replace('\\u0421', 'С')
    text = text.replace('\\u0422', 'Т').replace('\\u0423', 'У').replace('\\u0424', 'Ф')
    text = text.replace('\\u0425', 'Х').replace('\\u0426', 'Ц').replace('\\u0427', 'Ч')
    text = text.replace('\\u0428', 'Ш').replace('\\u0429', 'Щ').replace('\\u042a', 'Ъ')
    text = text.replace('\\u042b', 'Ы').replace('\\u042c', 'Ь').replace('\\u042d', 'Э')
    text = text.replace('\\u042e', 'Ю').replace('\\u042f', 'Я').replace('\\u0430', 'а')
    text = text.replace('\\u0431', 'б').replace('\\u0432', 'в').replace('\\u0433', 'г')
    text = text.replace('\\u0434', 'д').replace('\\u0435', 'е').replace('\\u0436', 'ж')
    text = text.replace('\\u0437', 'з').replace('\\u0438', 'и').replace('\\u0439', 'й')
    text = text.replace('\\u043a', 'к').replace('\\u043b', 'л').replace('\\u043c', 'м')
    text = text.replace('\\u043d', 'н').replace('\\u043e', 'о').replace('\\u043f', 'п')
    text = text.replace('\\u0440', 'р').replace('\\u0441', 'с').replace('\\u0442', 'т')
    text = text.replace('\\u0443', 'у').replace('\\u0444', 'ф').replace('\\u0445', 'х')
    text = text.replace('\\u0446', 'ц').replace('\\u0447', 'ч').replace('\\u0448', 'ш')
    text = text.replace('\\u0449', 'щ').replace('\\u044a', 'ъ').replace('\\u044b', 'ы')
    text = text.replace('\\u044c', 'ь').replace('\\u044d', 'э').replace('\\u044e', 'ю')
    text = text.replace('\\u044f', 'я')
    
    # Arabic characters - generic replacement for common encoding issues
    text = text.replace('\\u0627', 'ا').replace('\\u064a', 'ي').replace('\\u0644', 'ل')
    text = text.replace('\\u062a', 'ت').replace('\\u0646', 'ن').replace('\\u0633', 'س')
    text = text.replace('\\u0645', 'م').replace('\\u0631', 'ر').replace('\\u0648', 'و')
    text = text.replace('\\u0639', 'ع').replace('\\u062f', 'د').replace('\\u0628', 'ب')
    text = text.replace('\\u0649', 'ى').replace('\\u0629', 'ة').replace('\\u062c', 'ج')
    text = text.replace('\\u0642', 'ق').replace('\\u0641', 'ف').replace('\\u062d', 'ح')
    text = text.replace('\\u0635', 'ص').replace('\\u0637', 'ط').replace('\\u0632', 'ز')
    text = text.replace('\\u0634', 'ش').replace('\\u063a', 'غ').replace('\\u062e', 'خ')
    text = text.replace('\\u0623', 'أ').replace('\\u0621', 'ء').replace('\\u0624', 'ؤ')
    text = text.replace('\\u0626', 'ئ').replace('\\u0625', 'إ').replace('\\u0651', 'ّ')
    text = text.replace('\\u0652', 'ْ').replace('\\u064b', 'ً').replace('\\u064c', 'ٌ')
    text = text.replace('\\u064d', 'ٍ').replace('\\u064f', 'ُ').replace('\\u0650', 'ِ')
    text = text.replace('\\u064e', 'َ').replace('\\u0653', 'ٓ').replace('\\u0654', 'ٔ')
    text = text.replace('\\u0670', 'ٰ').replace('\\u0671', 'ٱ').replace('\\u0672', 'ٲ')
    text = text.replace('\\u0673', 'ٳ').replace('\\u0675', 'ٵ').replace('\\u0676', 'ٶ')
    text = text.replace('\\u0677', 'ٷ').replace('\\u0678', 'ٸ').replace('\\u0679', 'ٹ')
    text = text.replace('\\u067a', 'ٺ').replace('\\u067b', 'ٻ').replace('\\u067c', 'ټ')
    text = text.replace('\\u067d', 'ٽ').replace('\\u067e', 'پ').replace('\\u067f', 'ٿ')
    text = text.replace('\\u0680', 'ڀ').replace('\\u0681', 'ځ').replace('\\u0682', 'ڂ')
    text = text.replace('\\u0683', 'ڃ').replace('\\u0684', 'ڄ').replace('\\u0685', 'څ')
    text = text.replace('\\u0686', 'چ').replace('\\u0687', 'ڇ').replace('\\u0688', 'ڈ')
    text = text.replace('\\u0689', 'ډ').replace('\\u068a', 'ڊ').replace('\\u068b', 'ڋ')
    text = text.replace('\\u068c', 'ڌ').replace('\\u068d', 'ڍ').replace('\\u068e', 'ڎ')
    text = text.replace('\\u068f', 'ڏ').replace('\\u0690', 'ڐ').replace('\\u0691', 'ڑ')
    text = text.replace('\\u0692', 'ڒ').replace('\\u0693', 'ړ').replace('\\u0694', 'ڔ')
    text = text.replace('\\u0695', 'ڕ').replace('\\u0696', 'ږ').replace('\\u0697', 'ڗ')
    text = text.replace('\\u0698', 'ژ').replace('\\u0699', 'ڙ').replace('\\u069a', 'ښ')
    text = text.replace('\\u069b', 'ڛ').replace('\\u069c', 'ڜ').replace('\\u069d', 'ڝ')
    text = text.replace('\\u069e', 'ڞ').replace('\\u069f', 'ڟ').replace('\\u06a0', 'ڠ')
    text = text.replace('\\u06a1', 'ڡ').replace('\\u06a2', 'ڢ').replace('\\u06a3', 'ڣ')
    text = text.replace('\\u06a4', 'ڤ').replace('\\u06a5', 'ڥ').replace('\\u06a6', 'ڦ')
    text = text.replace('\\u06a7', 'ڧ').replace('\\u06a8', 'ڨ').replace('\\u06a9', 'ک')
    text = text.replace('\\u06aa', 'ڪ').replace('\\u06ab', 'ګ').replace('\\u06ac', 'ڬ')
    text = text.replace('\\u06ad', 'ڭ').replace('\\u06ae', 'ڮ').replace('\\u06af', 'گ')
    text = text.replace('\\u06b0', 'ڰ').replace('\\u06b1', 'ڱ').replace('\\u06b2', 'ڲ')
    text = text.replace('\\u06b3', 'ڳ').replace('\\u06b4', 'ڴ').replace('\\u06b5', 'ڵ')
    text = text.replace('\\u06b6', 'ڶ').replace('\\u06b7', 'ڷ').replace('\\u06b8', 'ڸ')
    text = text.replace('\\u06b9', 'ڹ').replace('\\u06ba', 'ں').replace('\\u06bb', 'ڻ')

    return text




def print_summary_points(data):

    if 'summary' in data:
        for point in data['summary']:
            point = fix_unicode(point)
            print(f"• {point}")

        print(f"\nTotal points: {data.get('Total', 'N/A')}")
        print(f"Recommended tone: {data.get('Tone', 'N/A')}")


def text_to_speech(text, output_file, language):
    # Map language codes
    language_map = {
        'anglais': 'en',
        'francais': 'fr',
        'espagnol': 'es',
        'arabe': 'ar',
        'allemand': 'de',
        'russe': 'ru',
        'italien': 'it',
        'portugais': 'pt'
    }
    
    # Get the correct language code
    lang_code = language_map.get(language.lower(), 'en')

    text = fix_unicode(text)
    
    try:
        # Initialize gTTS with text and language
        tts = gTTS(text=text, lang=lang_code, slow=False)
        
        # Save to file
        tts.save(output_file)
        print(f"Audio content saved to {output_file}")
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")


def generate_image(phrase, output_file):
    """Génère une image de haute qualité en fonction de la phrase donnée et du contexte marocain"""
    
    # Prompt amélioré pour une image réaliste et marocaine
    prompt = f"""Generate a high-quality realistic image of {phrase}. 
The image should reflect real-world objects or scenes, like modern Moroccan coins, contemporary Moroccan architecture, or everyday objects described in the phrase.
The style should be photographic, detailed, and suitable for a media publication. No text or logos should appear in the image.
The image should be true to reality, with sharp details, realistic textures, and natural lighting that mimics the Moroccan sunlight, focusing on the current and vibrant aspects of Morocco today."""

    # Envoie de la requête à DALL-E 3 pour générer l'image
    try:
        response = openai.Image.create(
            model="dall-e-3",  
            prompt=prompt,
            size="1024x1024",  
            n=1
        )

        # Vérifier si la génération de l'image a réussi
        if response and response['data']:
            image_url = response['data'][0]['url']
            
            # Télécharger l'image
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                # Convertir la réponse en image avec PIL
                image_data = Image.open(BytesIO(image_response.content))
                
                # Redimensionner l'image pour garder le ratio et enregistrer
                target_width = 1080
                target_height = 1920
                original_aspect = image_data.width / image_data.height
                target_aspect = target_width / target_height

                if original_aspect > target_aspect:
                    new_width = int(target_height * original_aspect)
                    new_height = target_height
                    image_data = image_data.resize((new_width, new_height))
                    left = (new_width - target_width) // 2
                    image_data = image_data.crop((left, 0, left + target_width, target_height))
                else:
                    new_height = int(target_width / original_aspect)
                    new_width = target_width
                    image_data = image_data.resize((new_width, new_height))
                    top = (new_height - target_height) // 2
                    image_data = image_data.crop((0, top, target_width, top + target_height))

                # Sauvegarder l'image générée
                image_data.save(output_file, format='JPEG')
                print(f"Image saved to {output_file}")
            else:
                print(f"Error downloading image: {image_response.status_code}")
        else:
            print("No image URL found in response.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def add_text_to_image(text, image_path, output_path):

    text = fix_unicode(text)
    
    # Open the image
    img = Image.open(image_path)
    
    # Create a copy of the image
    img_with_overlay = img.copy()
    
    # Create draw object
    draw = ImageDraw.Draw(img_with_overlay)
    
    # Calculate dimensions
    width, height = img.size
    
    # Add semi-transparent gray overlay over the entire image (with reduced opacity)
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 0), (width, height)], 
                           fill=(50, 50, 50, 140))  # Semi-transparent gray with reduced opacity
    
    # Paste overlay onto the image
    img_with_overlay = Image.alpha_composite(img_with_overlay.convert('RGBA'), overlay)
    
    # Add logo overlay
    try:
        logo = Image.open("logo.png")
        # Resize logo to be x of image width, maintaining aspect ratio
        logo_width = int(width * 0.4)  # percentage of image width
        logo_height = int(logo.height * (logo_width / logo.width))
        logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        
        # Calculate position to center logo at the top with some margin
        logo_x = (width - logo_width) // 2
        logo_y = int(height * 0.10)  # distance from the top
        
        # Make sure logo has alpha channel for proper overlay
        if logo.mode != 'RGBA':
            logo = logo.convert('RGBA')
            
        # Paste logo onto the image
        img_with_overlay.paste(logo, (logo_x, logo_y), logo)
    except Exception as e:
        print(f"Could not add logo: {e}")

    # Calculate text size and position - use a fixed larger font size for Streamlit
    font_size = max(int(width * 0.06), 48)  # Minimum size of 48 points, scales with width
    
    try:
        # Try to use a bold font if available
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                # Try system fonts that are commonly available
                font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        # If all else fails, use default but make it larger
        default_font = ImageFont.load_default()
        font = default_font.font_variant(size=font_size)
    
    # Wrap text to fit image width - fewer characters per line for better readability
    chars_per_line = max(15, int((width * 0.7) / (font_size * 0.6)))
    wrapped_text = textwrap.fill(text, width=chars_per_line)
    
    # Ensure text is properly encoded for handling non-Latin characters (Arabic, French, Spanish)
    # Calculate position to center text in the image
    draw = ImageDraw.Draw(img_with_overlay)
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (width - text_width) / 2
    # Adjust y position based on text_height to ensure text is properly centered vertically
    y = height * 0.6 - (text_height / 2)  # Position text in the lower third of the image
    
    # Draw thicker black outline for better visibility
    outline_thickness = max(3, int(font_size / 15))
    for offset in range(-outline_thickness, outline_thickness + 1):
        for offset2 in range(-outline_thickness, outline_thickness + 1):
            if abs(offset) + abs(offset2) <= outline_thickness + 1:  # Create rounded corners
                draw.text((x + offset, y + offset2), wrapped_text, font=font, fill='black')
    
    # Draw white text
    draw.text((x, y), wrapped_text, font=font, fill='white')
    
    # Save the image
    img_with_overlay.convert('RGB').save(output_path)
    print(f"collage saved to {output_path}")





def image_audio_to_video(image_dir, audio_dir, output_path, add_voiceover, add_music):
        
        # Get all image and audio files
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')))

        if not image_files or not audio_files:
            raise ValueError("No image or audio files found")


        # Check if we should include audio
        if not add_voiceover:  # controlling voiceover
            # Create clips without audio
            clips = []
            for image_file in image_files:
                # Get duration from matching audio file (same index)
                i = image_files.index(image_file)
                if i < len(audio_files):
                    audio = AudioFileClip(audio_files[i])
                    duration = audio.duration
                    audio.close()
                else:
                    duration = 3.0  # Default duration if no matching audio
                
                # Create image clip with no audio
                image = ImageClip(image_file).set_duration(duration)
                clips.append(image)

        else:
            # Create video clips with audio
            clips = []
            for image_file, audio_file in zip(image_files, audio_files):
                # Load the audio to get its duration
                audio = AudioFileClip(audio_file)
                # Create image clip with duration matching the audio
                image = ImageClip(image_file).set_duration(audio.duration)
                # Combine image with audio
                video_clip = image.set_audio(audio)
                clips.append(video_clip)

         # Add outro if it exists
        try:
            if os.path.exists("outro.png"):
                # Open the outro image
                outro_img = Image.open("outro.png")
                
                # Resize outro to match video dimensions (1080x1920)
                outro_img = outro_img.resize((1080, 1920), Image.Resampling.LANCZOS)
                
                # Save the resized outro temporarily
                outro_resized_path = "cache/img/outro_resized.jpg"
                outro_img.convert('RGB').save(outro_resized_path)
                
                # Create a silent audio clip for the outro (3 seconds)
                silent_audio = AudioClip(lambda t: 0, duration=3)
                
                # Create outro clip
                outro_clip = ImageClip(outro_resized_path).set_duration(3).set_audio(silent_audio)
                
                # Add outro to clips list
                clips.append(outro_clip)
                print("Added outro to video")
        except Exception as e:
            print(f"Failed to add outro: {e}")


        # Concatenate all clips
        final_video = concatenate_videoclips(clips)
        
        # Add background music if requested
        if add_music and os.path.exists("music.mp3"):
            try:
                # Load background music
                background_music = AudioFileClip("music.mp3")
                
                # Loop music if it's shorter than the video
                if background_music.duration < final_video.duration:
                    background_music = background_music.loop(duration=final_video.duration)
                else:
                    # Trim if longer than video
                    background_music = background_music.subclip(0, final_video.duration)
                
                # Reduce volume of background music
                background_music = background_music.volumex(0.3)
                
                if add_voiceover:
                    # Mix background music with existing audio
                    final_audio = CompositeAudioClip([final_video.audio, background_music])
                    final_video = final_video.set_audio(final_audio)
                else:
                    # Just use background music
                    final_video = final_video.set_audio(background_music)
                    
                print("Added background music")
            except Exception as e:
                print(f"Failed to add background music: {e}")
        
        # Write the final video
        final_video.write_videofile(output_path, fps=30)
        
        # Clean up
        final_video.close()
        for clip in clips:
            clip.close()

        print(f"Video saved to {output_path}")








def clear_cache():
    folders = ["cache/aud", "cache/img", "cache/clg"]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')




def do_work(data, language, add_voiceover, add_music):

    if 'summary' in data:
        i = 1
        for point in data['summary']:
            print(f"• {point}")
            os.makedirs("cache/aud/", exist_ok=True)
            text_to_speech(point, f"cache/aud/point_{i}.mp3", language)
            generate_image(point, f"cache/img/point_{i}.jpg")
            add_text_to_image(point, f"cache/img/point_{i}.jpg", f"cache/clg/point_{i}.jpg")
            i += 1
        image_audio_to_video("cache/clg", "cache/aud", f"cache/vid/final.mp4", add_voiceover, add_music)     




def main():
    st.title("Le Matin Ai Tools : Résumé d'article en vidéo 🔗📝🎬✨")
    
    # Create sidebar for settings
    st.sidebar.header("Settings")
    language = st.sidebar.selectbox(
        "Sélectionner la langue",
        ["Anglais", "Francais", "Espagnol", "Arabe", "Allemand", "Russe", "Italien", "Portugais"],
        index=1
    )
    
    slidenumber = st.sidebar.slider(
        "Nombre des Points et Idées",
        min_value=2,
        max_value=6,
        value=3
    )

    wordnumber = st.sidebar.slider(
        "Nombre des Mots par Point",
        min_value=10,
        max_value=20,
        value=13
    )

    # Add checkbox for music option
    add_music = st.sidebar.checkbox(
        "Ajouter une musique de fond",
        value=False,
        help="Ajouter une musique de fond à la vidéo générée"
    )

    # Add checkbox for VoiceOver option
    add_voiceover = st.sidebar.checkbox(
        "Ajouter une voix off",
        value=True,
        help="Ajouter une voix off à la vidéo générée"
    )

    
    # Main content area
    
    st.subheader("Choisissez comment fournir le contenu de votre article :")
    input_method = st.radio("Sélectionnez la méthode d’entrée :", ["Entrez un URL", "Écrire/Coller le texte de l’article"])
    article_text_input = " "
    url = " "
    
    if input_method == "Entrez un URL":
        url = st.text_input("Entrez l'URL de l’article:")
        article_text_input = None
        
    else:
        article_text_input = st.text_area("Écrire ou Coller le texte de l’article :", height=300)
        url = None  # Set URL to None when using direct text input
    
    # Make sure article_text is defined in both cases
    if st.button("Créer un résumé"):
            with st.spinner("Traitement de l’article..."):
                if input_method == "Entrez un URL":  # If URL is provided
                    if not url or url.strip() == "":
                        st.error("Veuillez fournir une URL..")
                        st.stop()
                    article_text = scrape_text_from_url(url)
                    st.success("Article récupéré avec succès !")
                else:  # If direct text input is provided
                    if not article_text_input or article_text_input.strip() == "":
                        st.error("Veuillez fournir du text.")
                        st.stop()
                    article_text = article_text_input
                    st.success("Article traité avec succès !")

            with st.spinner("Génération du résumé..."):
                llm_response = call_llm_api(article_text, slidenumber, wordnumber, language)
                Json = save_and_clean_json(llm_response, "summary.json")
                st.success("Résumé généré avec succès !")

            if 'summary' in Json:
                for point in Json['summary']:
                    point = fix_unicode(point)
                    st.write(f"• {point}")
                try:
                        with st.spinner("Génération du video..."):
                            do_work(Json, language, add_voiceover, add_music)
                            st.success("Video généré avec succès !")
            
                        # Display the generated video
                        if os.path.exists("cache/vid/final.mp4"):
                            st.video("cache/vid/final.mp4")
                            
                            # Add download button
                            with open("cache/vid/final.mp4", "rb") as file:
                                st.download_button(
                                    label="Télécharger la vidéo",
                                    data=file,
                                    file_name="generated_video.mp4",
                                    mime="video/mp4"
                                )
                        
                        # Clear cache after successful generation
                        clear_cache()
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("cache/aud/", exist_ok=True)
    os.makedirs("cache/img/", exist_ok=True)
    os.makedirs("cache/clg/", exist_ok=True)
    os.makedirs("cache/vid/", exist_ok=True)
    
    main()