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

st.set_page_config(page_title="АЯЛГУУ AI (MVP)", page_icon="🎵", layout="wide")

st.title("🎵 АЯЛГУУ AI — Хувийн Туршилтын Хувилбар")
st.markdown("""
Энэ бол "Аялгууг хадгалж, зөвхөн найруулгыг өөрчлөх" санааг турших хөнгөн MVP юм.  
**Ажиллах зарчим:** Дуулсан/хэмссэн аудио → Аялгуу (Pitch) таних → MIDI болгох → "Ая түгжих".
""")

st.header("1. Аялгуугаа оруулна уу")
audio_file = st.file_uploader("Аудио файл (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())
    
    st.audio(audio_file, format="audio/wav")
    st.success("✅ Аудио уншигдлаа. Одоо аялгууг шинжилж байна...")
    
    with st.spinner("🎼 Librosa pYIN алгоритмаар аялгууг нот болгон хувиргаж байна..."):
        try:
            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            times = librosa.times_like(f0, sr=sr)
            valid_idx = ~np.isnan(f0)
            valid_times = times[valid_idx]
            valid_f0 = f0[valid_idx]
            valid_midi_notes = librosa.hz_to_midi(valid_f0)
            
            st.success("🎉 Аялгуу амжилттай танигдлаа!")
            
            st.subheader("📊 Танигдсан аялгууны график (Pitch Contour)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(valid_times, valid_midi_notes, label='Танигдсан нот (MIDI)', color='#4CAF50', alpha=0.8)
            ax.set_xlabel('Хугацаа (секунд)')
            ax.set_ylabel('MIDI Нотын дугаар')
            ax.set_title('Аялгууны хэлбэр (Melody Contour)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("💾 Аялгууг MIDI болгон хадгалах (Ая түгжих)")
            st.info("Энэ MIDI файл нь таны аялгууг 'бэлгэдлийн түвшинд' (symbolic) түгжсэн хэлбэр юм.")
            
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
            
            mid = create_midi(valid_times, valid_midi_notes)
            midi_bytes = io.BytesIO()
            mid.save(midi_bytes)
            midi_bytes.seek(0)
            
            st.download_button(
                label="️ Миний Аялгуу (MIDI) татаж авах",
                data=midi_bytes,
                file_name="ayalguu_locked.mid",
                mime="audio/midi"
            )
            
            st.markdown("---")
            st.subheader("🚀 Дараагийн алхам (Төлөвлөгөө)")
            if st.button("🎹 Найруулга симуляци хийх (Туршилт)"):
                with st.spinner("AI найруулга хийж байна..."):
                    time.sleep(2)
                    st.success("✅ Симуляци амжилттай! Бодит хувилбарт эндээс MIDI-г дуу болгон хувиргана.")
                    
        except Exception as e:
            st.error(f"Аудио шинжлэх явцад алдаа гарлаа: {e}")
            st.write("💡 Зөвлөмж: Цэвэр, зөвхөн нэг хоолойтой (дуулалт эсвэл хэмсэлт) богино аудио ашиглавал илүү сайн ажиллана.")
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
