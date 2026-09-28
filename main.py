import os
import tempfile
import threading
import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from tensorflow.keras.models import load_model

from src.data_preprocessing import extract_frames, SEQUENCE_LENGTH
from src.feature_extraction import extract_video_keypoints
from src.predict import text_to_speech
from deep_translator import MyMemoryTranslator
from deep_translator.exceptions import TooManyRequests
from src.services.sentence_refiner import refine_sentence
from src.services.translation_cache import TranslationCache

app = FastAPI(title="Sign Language Recognition API")

# Initialize Cache
translation_cache = TranslationCache()
TRANSLATION_LOCK = threading.Lock()
TRANSLATION_INTERVAL_SECONDS = 0.5
_last_translation_at = 0.0

# MyMemoryTranslator uses locale codes (e.g. 'hi-IN') rather than simple ISO codes.
# This mapping converts the gTTS/app language codes stored in SUPPORTED_LANGUAGES to
# the locale codes that MyMemoryTranslator accepts.  The gTTS codes are kept as the
# canonical identifiers throughout the app (TM keys, /languages endpoint, TTS calls).
MYMEMORY_LANG_MAP = {
    # Indian languages
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "ur": "ur-PK",
    # International languages
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "it": "it-IT",
    "pt": "pt-PT",
    "nl": "nl-NL",
    "ru": "ru-RU",
    "tr": "tr-TR",
    "ar": "ar-SA",
    "zh-CN": "zh-CN",
    "ja": "ja-JP",
    "ko": "ko-KR",
}

SUPPORTED_LANGUAGES = {
    "Indian Languages": {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Urdu": "ur"
    },
    "International Languages": {
        "French": "fr",
        "German": "de",
        "Spanish": "es",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Turkish": "tr",
        "Arabic": "ar",
        "Chinese": "zh-CN",
        "Japanese": "ja",
        "Korean": "ko"
    }
}

LANGUAGES = {}
for category, langs in SUPPORTED_LANGUAGES.items():
    LANGUAGES.update(langs)


def translate_text(text: str, target_lang_code: str) -> str:
    """
    Translate *text* (English) to *target_lang_code* using MyMemoryTranslator.

    Requests are serialised through a lock so rapid language switching does not
    hammer the free MyMemory API tier.  Five retries with exponential back-off
    are attempted before the exception is re-raised to the caller.

    Parameters
    ----------
    text : str
        English source text.
    target_lang_code : str
        gTTS/app language code (e.g. 'hi', 'fr', 'zh-CN').  The function maps
        this to the MyMemory locale code internally.

    Returns
    -------
    str
        Translated text.
    """
    global _last_translation_at

    mymemory_target = MYMEMORY_LANG_MAP.get(target_lang_code)
    if not mymemory_target:
        raise ValueError(
            f"No MyMemory locale code found for language code '{target_lang_code}'. "
            f"Supported codes: {list(MYMEMORY_LANG_MAP.keys())}"
        )

    with TRANSLATION_LOCK:
        elapsed = time.monotonic() - _last_translation_at
        if elapsed < TRANSLATION_INTERVAL_SECONDS:
            time.sleep(TRANSLATION_INTERVAL_SECONDS - elapsed)

        for attempt in range(5):
            _last_translation_at = time.monotonic()
            try:
                translator = MyMemoryTranslator(source="en-US", target=mymemory_target)
                result = translator.translate(text)
                if not result:
                    raise ValueError("MyMemoryTranslator returned empty translation.")
                return result
            except TooManyRequests:
                # MyMemory free tier: 500 words/day without a key; raise immediately.
                raise
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise Exception("Translation failed after 5 attempts")

@app.get("/languages")
async def get_languages():
    return SUPPORTED_LANGUAGES

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static directory for audio
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables to hold model artifacts
MODEL = None
CLASSES = None
NORM_MEAN = None
NORM_STD = None

@app.on_event("startup")
async def load_artifacts():
    global MODEL, CLASSES, NORM_MEAN, NORM_STD
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    model_path = os.path.join(models_dir, "sign_language_model.h5")
    classes_path = os.path.join(models_dir, "label_encoder.npy")
    norm_path = os.path.join(models_dir, "norm_stats.npz")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return
        
    MODEL = load_model(model_path)
    CLASSES = np.load(classes_path, allow_pickle=True)
    norm = np.load(norm_path)
    NORM_MEAN = norm['mean']
    NORM_STD = norm['std']
    print("Model artifacts loaded successfully.")

@app.post("/predict")
async def predict_video(video: UploadFile = File(...), language: str = Form("English")):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train the model first.")
        
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        # Extract frames
        frames = extract_frames(tmp_path, SEQUENCE_LENGTH)
        if frames is None:
            raise HTTPException(status_code=400, detail="Could not read the video. Please upload a valid file.")

        # Extract keypoints
        keypoints = extract_video_keypoints(frames)

        # Normalize
        keypoints = (keypoints - NORM_MEAN) / (NORM_STD + 1e-8)

        # Predict
        X = keypoints[np.newaxis, ...]
        probs = MODEL.predict(X, verbose=0)[0]
        predicted_idx = int(np.argmax(probs))
        predicted_label = str(CLASSES[predicted_idx])
        confidence = float(probs[predicted_idx])

        # Refine Sentence
        refined_prediction = refine_sentence(predicted_label)

        # Translation
        lang_code = LANGUAGES.get(language, "en")
        translated_text = refined_prediction
        translation_source = "cache"

        if lang_code != "en":
            cached_trans = translation_cache.get(refined_prediction, lang_code)
            if cached_trans:
                translated_text = cached_trans
                translation_source = "cache"
            else:
                translation_source = "translator"
                try:
                    translated_text = translate_text(refined_prediction, lang_code)
                    translation_cache.set(refined_prediction, lang_code, translated_text)
                except Exception as e:
                    print(f"[ERROR] Translation failed for lang_code='{lang_code}': {e}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Translation to '{language}' failed: {e}",
                    ) from e

        # Text to Speech
        audio_filename = f"output_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        text_to_speech(translated_text, audio_path, lang=lang_code)
        
        # Cleanup temp video file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        return {
            "original_prediction": predicted_label,
            "refined_prediction": refined_prediction,
            "translated_text": translated_text,
            "selected_language": language,
            "language_code": lang_code,
            "translation_source": translation_source,
            "confidence": confidence,
            "audio_url": f"/static/audio/{audio_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text_endpoint(text: str = Form(...), language: str = Form("English")):
    try:
        lang_code = LANGUAGES.get(language, "en")
        translated_text = text
        translation_source = "cache"

        if lang_code != "en":
            cached_trans = translation_cache.get(text, lang_code)
            if cached_trans:
                translated_text = cached_trans
                translation_source = "cache"
            else:
                translation_source = "translator"
                try:
                    translated_text = translate_text(text, lang_code)
                    translation_cache.set(text, lang_code, translated_text)
                except TooManyRequests as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Translation service is rate-limited. Please try again shortly.",
                    ) from exc
                except Exception as exc:
                    print(f"[ERROR] Translation failed for lang_code='{lang_code}': {exc}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Translation to '{language}' failed: {exc}",
                    ) from exc

        # Text to Speech
        audio_filename = f"output_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        text_to_speech(translated_text, audio_path, lang=lang_code)

        return {
            "original_text": text,
            "translated_text": translated_text,
            "selected_language": language,
            "language_code": lang_code,
            "translation_source": translation_source,
            "audio_url": f"/static/audio/{audio_filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

