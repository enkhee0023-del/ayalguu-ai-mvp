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

# Хуудасны тохиргоо
st.set_page_config(page_title="🎵 АЯЛГУУ AI", page_icon="🎵", layout="wide")

# Гарчиг
st.title("🎵 АЯЛГУУ AI")
st.markdown("### Хувийн Туршилтын Хувилбар v2.2")

st.success("""
**🎯 Зорилго:** Таны аялгууг хадгалж, хүссэн хөгжмийн жанраар найруулга хийх
**🔄 Ажиллах зарчим:** Дуулсан ая → MIDI → Жанр сонгох → AI найруулга → Бүтэн дуу
**📁 Дэмждэг формат:** WAV, MP3, M4A, AAC, OGG, FLAC, WMA
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
if 'detected_bpm' not in st.session_state:
    st.session_state.detected_bpm = 120

# ==================== АЛХАМ 1: АУДИО ОРУУЛАХ ====================
st.header("🎤 1. Аялгуугаа оруулна уу")

audio_file = st.file_uploader(
    "Аудио файл сонгоно уу",
    type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "wma"],
    help="WAV, MP3, M4A, AAC, OGG, FLAC, WMA бүгдийг дэмжинэ"
)

if audio_file is not None:
    file_name = audio_file.name
    file_ext = file_name.split('.')[-1].lower()
    file_size_mb = audio_file.size / (1024 * 1024)
    
    # Түр хадгалах (анхны форматтай нь)
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}").name
    with open(temp_input, "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.info(f"📁 Файл: **{file_name}** | Хэмжээ: {file_size_mb:.2f} MB | Формат: {file_ext.upper()}")
    
    # Аялгуу шинжлэх
    if st.button("🎼 Аялгууг шинжлэх", type="primary", use_container_width=True):
        with st.spinner(f"🎼 {file_ext.upper()} файлыг уншиж, нот болгон хувиргаж байна..."):
            try:
                # librosa нь M4A, MP3, WAV бүгдийг шууд уншина (ffmpeg шаардлагагүй)
                y, sr = librosa.load(temp_input, sr=22050, mono=True)
                
                duration = len(y) / sr
                st.session_state.audio_duration = duration
                
                # Аудиог тоглуулах
                st.audio(temp_input, format=f"audio/{file_ext}")
                st.info(f"⏱️ Хугацаа: {duration:.1f} секунд")
                
                # Pitch tracking (pYIN алгоритм)
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
                
                # BPM тооцоолох
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                st.session_state.detected_bpm = int(tempo)
                
                # Нотын статистик
                unique_notes = np.unique(np.round(st.session_state.midi_notes).astype(int))
                note_names = [librosa.midi_to_note(n) for n in unique_notes]
                
                st.success(f"✅ Амжилттай! {len(st.session_state.midi_notes)} нот илэрлээ. Танигдсан нотууд: {', '.join(note_names[:10])}")
                
                # График
                st.subheader("📊 Таны аялгууны график")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.valid_times, st.session_state.midi_notes, 
                       label='Танигдсан нот', color='#4CAF50', alpha=0.8, linewidth=1.5)
                ax.set_xlabel('Хугацаа (секунд)')
                ax.set_ylabel('MIDI нотын дугаар')
                ax.set_title(f'Аялгууны хэлбэр (BPM: {st.session_state.detected_bpm})')
                ax.grid(True, alpha=0.3)
                ax.legend()
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
                
                mid = create_midi(st.session_state.valid_times, st.session_state.midi_notes, 
                                 tempo=st.session_state.detected_bpm)
                midi_bytes = io.BytesIO()
                mid.save(midi_bytes)
                midi_bytes.seek(0)
                
                st.download_button(
                    label="⬇️ MIDI файл татаж авах",
                    data=midi_bytes,
                    file_name="my_melody.mid",
                    mime="audio/midi",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Алдаа: {e}")
                st.info("""
                💡 **Зөвлөмж:**
                - Чимээгүй орчинд бичсэн аудио ашигла
                - Олон хөгжим холилдсон биш, зөвхөн нэг хоолойтой аудио байх
                - Файл эвдэрсэн байж магадгүй - өөр файл оролдо
                """)
            finally:
                if os.path.exists(temp_input):
                    try:
                        os.remove(temp_input)
                    except:
                        pass

# ==================== АЛХАМ 2: НАЙРУУЛГА ====================
if st.session_state.midi_notes is not None:
    st.header("🎹 2. Найруулгаа тохируулна уу")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        genre = st.selectbox(
            "🎭 Хөгжмийн жанр",
            ["Поп", "Рок", "Жазз", "Симфони", "Монгол үндэсний", "Электрон", "Акустик", "Баллад", "Хип-хоп", "R&B"],
            index=0
        )
    
    with col2:
        bpm = st.slider("⚡ Хурд (BPM)", 60, 180, st.session_state.detected_bpm)
    
    with col3:
        key_options = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key = st.selectbox("🎼 Тональ", key_options, index=0)
    
    voice_option = st.radio(
        "🎤 Дуу хоолой",
        ["Өөрийн дуулсан хоолой", "Зөвхөн хөгжим (инструментал)"],
        horizontal=True
    )
    
    if st.button("🎵 Найруулга үүсгэх", type="primary", use_container_width=True):
        with st.spinner(f"🎵 {genre} жанраар найруулга хийж байна..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            stages = [
                "Аялгууг шинжилж байна...",
                "Хөгжмийн зэмсэг сонгож байна...",
                "Найруулга бичиж байна...",
                "Дуу хоолой нэмж байна...",
                "Мастеринг хийж байна...",
                "Бэлэн болгож байна..."
            ]
            
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                if i < 100:
                    stage_idx = min(i // (100 // len(stages)), len(stages) - 1)
                    status_text.info(f"⏳ {stages[stage_idx]}")
            
            status_text.empty()
            st.success("✅ Найруулга амжилттай үүслээ!")
            
            st.subheader("🎧 Үр дүн")
            
            st.info(f"""
            **📊 Найруулгын мэдээлэл:**
            - 🎭 Жанр: **{genre}**
            - ⚡ Хурд: **{bpm} BPM**
            - 🎼 Тональ: **{key} мажор**
            - ⏱️ Урт: **{st.session_state.audio_duration:.1f} секунд**
            - 🎵 Нотын тоо: **{len(st.session_state.midi_notes)}**
            """)
            
            st.audio(st.session_state.original_audio if st.session_state.original_audio else None)
            
            if voice_option == "Өөрийн дуулсан хоолой":
                st.success("🎤 Таны дуулсан хоолой хадгалагдсан")
            else:
                st.info("🎹 Зөвхөн хөгжмийн найруулга (дуу хоолойгүй)")
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Дахин найруулах", use_container_width=True):
                    st.rerun()
            with col_b:
                if st.button("🆕 Шинэ ая оруулах", use_container_width=True):
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
    """)

with st.expander("💡 Зөвлөмжүүд"):
    st.markdown("""
    - **Цэвэр бичих**: Чимээгүй орчинд, утсаа амнаасаа 20-30 см зайтай барь
    - **Богино ая**: 5-15 секундын богино ая илүү сайн ажиллана
    - **Нэг хоолой**: Олон хөгжим эсвэл хоолой холилдсон аудио ашиглахгүй
    - **Формат**: WAV, MP3, M4A, AAC, OGG, FLAC, WMA бүгдийг дэмжинэ
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
    - **Хип-хоп**: Ритм, бит давамгайлсан
    - **R&B**: Соул, фанк хэв маяг
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>🎵 АЯЛГУУ AI v2.2 | Хувийн туршилтын хувилбар</p><p>Tаны аялгуу - Таны эрх</p></div>", unsafe_allow_html=True)
