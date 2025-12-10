import io
import os
import base64
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
if not os.path.exists('static'):
    os.makedirs('static')
app.mount('/', StaticFiles(directory='static', html=True), name='static')


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    return data


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()

        # Write uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file.filename.split('.')[-1]) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Load audio
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)

        # -----------------------------
        # TEMPO ANALYSIS
        # -----------------------------
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        bpm_bins = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
        local_bpms = (tempogram * bpm_bins[:, None]).sum(axis=0) / (tempogram.sum(axis=0) + 1e-6)
        local_bpms = np.nan_to_num(local_bpms)

        # -----------------------------
        # PITCH ANALYSIS (YIN)
        # -----------------------------
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        times = librosa.times_like(f0, sr=sr)

        safe_f0 = np.where(f0 > 0, f0, np.nan)
        midi = 69 + 12 * np.log2(safe_f0 / 440.0)
        nearest = np.round(midi)
        cents_error = (midi - nearest) * 100
        avg_cents = float(np.nanmean(np.abs(cents_error)))
        median_cents = float(np.nanmedian(np.abs(cents_error)))

        # -----------------------------
        # PLOTS
        # -----------------------------

        # Pitch plot
        fig1 = plt.figure(figsize=(10, 3))
        plt.plot(times, safe_f0, linewidth=1)
        plt.title("Estimated fundamental frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.ylabel("Hz")
        img1 = fig_to_base64(fig1)

        # Pitch error plot
        fig2 = plt.figure(figsize=(10, 3))
        plt.plot(times, cents_error, linewidth=1)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title("Pitch error (cents)")
        plt.xlabel("Time (s)")
        plt.ylabel("Cents (±100)")
        plt.ylim(-100, 100)
        img2 = fig_to_base64(fig2)

        # Local tempo plot
        fig3 = plt.figure(figsize=(10, 3))
        if len(local_bpms) > 0:
            t_local = librosa.frames_to_time(np.arange(len(local_bpms)), sr=sr)
            plt.plot(t_local, local_bpms, linewidth=1)
            plt.title("Local estimated tempo (BPM)")
            plt.xlabel("Time (s)")
            plt.ylabel("BPM")
        img3 = fig_to_base64(fig3)

        # Summary
        result = {
            "tempo": float(tempo),
            "tempo_std": float(np.nanstd(local_bpms)),
            "avg_cents_error": avg_cents,
            "median_cents_error": median_cents,
            "images": {
                "f0_png_base64": img1,
                "cents_png_base64": img2,
                "local_bpm_png_base64": img3,
            }
        }

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except:
            pass

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
