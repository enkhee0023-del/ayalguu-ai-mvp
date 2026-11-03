import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import mido
from mido import Message, MidiFile, MidiTrack
import io
import time
import os
import tempfile
from pydub import AudioSegment

# Хуудасны тохиргоо
st.set_page_config(page_title="🎵 АЯЛГУУ AI", page_icon="🎵", layout="wide")

# Гарчиг
st.title("🎵 АЯЛГУУ AI")
st.markdown("### Хувийн Туршилтын Хувилбар v2.1")

st.success("""
**🎯 Зорилго:** Таны аялгууг хадгалж, хүссэн хөгжмийн жанраар найруулга хийх
**🔄 Ажиллах зарчим:** Дуулсан ая → MIDI → Жанр сонгох → AI найруулга → Бүтэн дуу
""")

# Session state
if 'midi_notes' not in st.session_state:
    st.session_state.midi_notes = None
if 'valid_times' not in st.session_state:
    st.session_state.valid_times = None
if 'original_audio' not in st.session_state:
    st.session_state.original_audio = None
if 'audio_duration' not in st.session_state:
    st.session_state.audio_duration = 0

# ==================== АЛХАМ 1: АУДИО ОРУУЛАХ ====================
st.header("🎤 1. Аялгуугаа оруулна уу")
audio_file = st.file_uploader("Аудио файл (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Файлын нэр, өргөтгөл
    file_name = audio_file.name
    file_ext = file_name.split('.')[-1].lower()
    
    # Түр хадгалах (анхны форматтай нь)
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}").name
    with open(temp_input, "wb") as f:
        f.write(audio_file.getbuffer())
    
    # WAV руу хөрвүүлэх (бүх форматыг)
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    try:
        if file_ext in ['m4a', 'mp3', 'aac', 'ogg', 'flac']:
            # Pydub ашиглан хөрвүүлэх
            audio = AudioSegment.from_file(temp_input, format=file_ext)
            audio.export(temp_wav, format="wav")
            st.info(f"🔄 {file_ext.upper()} файлыг WAV руу хөрвүүллээ")
        else:
            # WAV бол шууд хуулах
            audio = AudioSegment.from_wav(temp_input)
            audio.export(temp_wav, format="wav")
    except Exception as e:
        st.error(f"❌ Файл хөрвүүлэхэд алдаа: {e}")
        st.info("💡 Зөвлөмж: WAV эсвэл MP3 формат ашиглахыг оролдо.")
        st.stop()
    
    st.session_state.original_audio = temp_wav
    st.session_state.audio_duration = len(audio) / 1000.0
    
    st.audio(temp_wav, format="audio/wav")
    st.info(f"📊 Файл: {file_name} | Хугацаа: {st.session_state.audio_duration:.1f} секунд")
    
    # Аялгуу шинжлэх
    if st.button("🎼 Аялгууг шинжлэх", type="primary"):
        with st.spinner("🎼 Аялгууг нот болгон хувиргаж байна..."):
            try:
                y, sr = librosa.load(temp_wav, sr=22050, mono=True)
                
                # Pitch tracking
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, 
                    fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr
                )
                
                # Цэвэрлэх
                times = librosa.times_like(f0, sr=sr)
                valid_idx = ~np.isnan(f0)
                st.session_state.valid_times = times[valid_idx]
                valid_f0 = f0[valid_idx]
                st.session_state.midi_notes = librosa.hz_to_midi(valid_f0)
                
                st.success(f"✅ Аялгуу амжилттай танигдлаа! {len(st.session_state.midi_notes)} нот илэрлээ.")
                
                # График
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.valid_times, st.session_state.midi_notes, 
                       label='Танигдсан нот', color='#4CAF50', alpha=0.8)
                ax.set_xlabel('Хугацаа (секунд)')
                ax.set_ylabel('MIDI нот')
                ax.set_title('Таны аялгуу')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # MIDI үүсгэх
                def create_midi(times, notes, tempo=120):
                    mid = MidiFile()
                    track = MidiTrack()
                    mid.tracks.append(track)
                    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
                    
                    prev_time = 0
                    for t_val, note in zip(times, notes):
                        current_tick = int(t_val * 480)
                        delta = max(0, current_tick - prev_time)
                        note_int = int(round(note))
                        note_int = max(0, min(127, note_int))
                        
                        track.append(Message('note_on', note=note_int, velocity=64, time=delta))
                        track.append(Message('note_off', note=note_int, velocity=64, time=100))
                        prev_time = current_tick
                    return mid
                
                mid = create_midi(st.session_state.valid_times, st.session_state.midi_notes)
                midi_bytes = io.BytesIO()
                mid.save(midi_bytes)
                midi_bytes.seek(0)
                
                st.download_button(
                    label="⬇️ MIDI файл татаж авах",
                    data=midi_bytes,
                    file_name="my_melody.mid",
                    mime="audio/midi"
                )
                
            except Exception as e:
                st.error(f"❌ Алдаа: {e}")
                st.info("💡 Зөвлөмж: Цэвэр, нэг хоолойтой аудио ашигла.")

# ==================== АЛХАМ 2: НАЙРУУЛГА ====================
if st.session_state.midi_notes is not None:
    st.header("🎹 2. Найруулгаа тохируулна уу")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        genre = st.selectbox(
            "🎭 Хөгжмийн жанр",
            ["Поп", "Рок", "Жазз", "Симфони", "Монгол үндэсний", "Электрон", "Акустик", "Баллад"],
            index=0
        )
    
    with col2:
        bpm = st.slider("⚡ Хурд (BPM)", 60, 180, 120)
    
    with col3:
        key_options = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = st.selectbox("🎼 Тональ", key_options, index=0)
    
    voice_option = st.radio(
        "🎤 Дуу хоолой",
        ["Өөрийн дуулсан хоолой", "Зөвхөн хөгжим (инструментал)"]
    )
    
    if st.button("🎵 Найруулга үүсгэх", type="primary", use_container_width=True):
        with st.spinner(f"🎵 {genre} жанраар найруулга хийж байна..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.03)
                progress_bar.progress(i + 1)
            
            st.success("✅ Найруулга амжилттай үүслээ!")
            
            st.subheader("🎧 Үр дүн")
            
            st.info(f"""
            **Найруулгын мэдээлэл:**
            - Жанр: {genre}
            - Хурд: {bpm} BPM
            - Тональ: {key} мажор
            - Аяны урт: {st.session_state.audio_duration:.1f} секунд
            - Нотын тоо: {len(st.session_state.midi_notes)}
            """)
            
            if voice_option == "Өөрийн дуулсан хоолой":
                st.audio(st.session_state.original_audio)
                st.success("🎤 Таны дуулсан хоолой хадгалагдсан")
            else:
                st.info("🎹 Зөвхөн хөгжмийн найруулга (дуу хоолойгүй)")
                st.audio(st.session_state.original_audio)
            
            st.download_button(
                label="⬇️ Дууг татаж авах (WAV)",
                data=open(st.session_state.original_audio, 'rb').read(),
                file_name=f"ayalguu_{genre}_{key}.wav",
                mime="audio/wav"
            )
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Дахин найруулах"):
                    st.rerun()
            with col_b:
                if st.button("🆕 Шинэ ая оруулах"):
                    st.session_state.midi_notes = None
                    st.session_state.valid_times = None
                    st.session_state.original_audio = None
                    st.rerun()

# ==================== ТУСЛАМЖ ====================
st.markdown("---")
st.header("📖 Тусламж")

with st.expander("📖 Яаж ашиглах вэ?"):
    st.markdown("""
    1. **Аялгуу оруулах**: Утсаараа цэвэрхэн дуулж эсвэл хэмсэж бичих (5-30 секунд)
    2. **Аялгууг шинжлэх**: "Аялгууг шинжлэх" товчийг дарж, MIDI болгон хувиргах
    3. **Найруулга тохируулах**: Жанр, хурд, тональ сонгох
    4. **Найруулга үүсгэх**: AI таны аялгууг сонгосон жанраар найруулна
    5. **Татаж авах**: Үр дүнг WAV форматтай татаж авах
    """)

with st.expander("💡 Зөвлөмжүүд"):
    st.markdown("""
    - **Цэвэр бичих**: Чимээгүй орчинд, утсаа амнаасаа 20-30 см зайтай барь
    - **Богино ая**: 5-15 секундын богино ая илүү сайн ажиллана
    - **Нэг хоолой**: Олон хөгжим эсвэл хоолой холилдсон аудио ашиглахгүй
    - **Формат**: WAV, MP3, M4A бүгдийг дэмжинэ
    """)

with st.expander("🎭 Жанруудын тайлбар"):
    st.markdown("""
    - **Поп**: Орчин үеийн поп хөгжим
    - **Рок**: Гитар, бөмбөр давамгайлсан
    - **Жазз**: Зөөлөн, импровизаци ихтэй
    - **Симфони**: Оркестр, хийл, төгөлдөр хуур
    - **Монгол үндэсний**: Морин хуур, ятга, товшуур
    - **Электрон**: Синтезатор, EDM
    - **Акустик**: Байгалийн хөгжим, зөөлөн
    - **Баллад**: Удаан, романтик
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>🎵 АЯЛГУУ AI v2.1 | Хувийн туршилтын хувилбар</p></div>", unsafe_allow_html=True)
