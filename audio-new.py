import queue
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import sounddevice as sd
import torch
import webrtcvad
from faster_whisper import WhisperModel

from pynput import keyboard
import threading

from rich import print
import json
import requests

WITHOUT_THINK = ['qwen3.5:9b']

MODEL_STRING = 'qwen3.5:9b'
# MODEL_STRING = 'qwen2.5:7b'
# MODEL_STRING = 'phi3:mini'


SYSTEM_PROMPT = """You will receive a noisy ASR transcript. It may contain:
- wrong punctuation, repetitions, omissions, homophones
- filler words, false starts, meaningless fragments

Do this in two steps:

Step 1 (Repair)
- Restore the intended meaning without adding new facts
- Remove noisy information and obvious mistakes, but keep uncertain words if they might be relevant
- Remove obvious repetitions and fillers
- Re-segment sentences clearly
- Preserve proper nouns, numbers, and acronyms
- If unsure, keep the original wording instead of guessing

Do not output for Step 1:
Cleaned Transcript: <cleaned transcript only>

Step 2 (Interpret & Answer)
- Extract the user's actual question, intent, or problem
- If the question seems like an interview question, answer in a clear and professional interview style
- Do not ask clarification questions first
- If something is uncertain, state it briefly instead of asking questions
- Infer the most likely meaning conservatively from the transcript
- Answer clearly, naturally, and briefly, with enough detail to be useful
- Focus on practical reasoning and real-world considerations
- Explain step by step only if necessary

Output for Step 2:
Answer: <the answer>
"""

# Remote call version (for Ollama/OpenAI API)
# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="ollama"
# )

# stream = client.chat.completions.create(
#     model=model,
#     temperature=0.0,
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"ASR_RAW:\n{asr_raw}\n\nPlease follow the 2 steps."}
#     ],
#     stream=True
# )

# answer = ""

# for chunk in stream:
#     delta = chunk.choices[0].delta
#     if delta.content:
#         print(delta.content, end="", flush=True)   
#         answer += delta.content

# print() 
# return answer


def asr_to_answer(asr_raw: str, model: str = MODEL_STRING) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ASR_RAW:\n{asr_raw}\n\nPlease follow the 2 steps."}
        ],
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0
        }
    }
    
    if MODEL_STRING in WITHOUT_THINK:
        payload['think'] = False
    

    answer_parts = []

    with requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        stream=True,
        timeout=600
    ) as r:
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue

            chunk = json.loads(line)
            message = chunk.get("message", {})
            content = message.get("content", "")

            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)

            if chunk.get("done", False):
                break

    print()
    return "".join(answer_parts)


class TranscriptBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._buf = ""

    def append(self, s: str):
        if not s:
            return
        with self._lock:

            if self._buf and not self._buf.endswith((" ", "\n")):
                self._buf += " "
            self._buf += s

    def dump_and_clear(self) -> str:
        with self._lock:
            out = self._buf.strip()
            self._buf = ""
            return out




def start_listener(get_text_fn):
    def on_press(key):
        try:
            if key == keyboard.Key.f12:
                text = get_text_fn()  
                print("\n========== [Question] ==========")
                print(f'{text}' if text else "(empty)")
                print("========== [Answer] ==========\n")
                if text:
                    asr_to_answer(text)
        except Exception as e:
            print("F12 listener error:", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener



# =======================
# Config
# =======================
@dataclass
class Config:
    # Audio
    frame_ms: int = 20
    sr_asr: int = 16000
    ch_try: int = 2

    # Sliding window
    hop_sec: float = 0.60
    window_sec: float = 2.40

    # VAD
    vad_mode: int = 3
    vad_check_sec: float = 0.40
    vad_speech_ratio: float = 0.35
    display_silence_ratio: float = 0.15
    silence_break_frames: int = 4

    # Whisper
    model_size: str = "base"
    language: str = "en"
    beam_size: int = 1

    # Seg filtering
    emit_guard_sec: float = 0.06
    min_seg_dur_sec: float = 0.10
    min_print_chars: int = 2

    # Sentence boundary
    max_sentence_sec: float = 4.0

    # Stable output gating
    pending_stable_required: int = 2
    pending_stable_min_chars: int = 28
    pending_similarity: float = 0.92

    # Ring buffer
    ring_extra_sec: float = 1.0
    ring_min_sec: float = 3.0


# =======================
# Utilities: audio
# =======================
def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    ratio = sr_out / sr_in
    n_out = max(1, int(len(x) * ratio))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


# =======================
# Device selection
# =======================
class DeviceSelector:
    @staticmethod
    def pick_wasapi_loopback_device(prefer_default_output: bool = True):
        hostapis = sd.query_hostapis()
        devices = sd.query_devices()

        wasapi_index = None
        for i, h in enumerate(hostapis):
            if "WASAPI" in h.get("name", ""):
                wasapi_index = i
                break
        if wasapi_index is None:
            raise RuntimeError("No WASAPI hostapi found. This is for Windows.")

        default_out = sd.default.device[1]
        default_out_name = None
        if isinstance(default_out, int) and 0 <= default_out < len(devices):
            default_out_name = devices[default_out]["name"].lower()

        candidates = []
        for idx, d in enumerate(devices):
            if d.get("hostapi") != wasapi_index:
                continue
            if d.get("max_input_channels", 0) <= 0:
                continue

            name_l = d.get("name", "").lower()
            if "loopback" in name_l:
                score = 0
                if prefer_default_output and default_out_name:
                    if any(tok in name_l for tok in default_out_name.split()[:4]):
                        score += 10
                if d.get("max_input_channels", 0) >= 2:
                    score += 2
                candidates.append((score, idx, d.get("name", ""), d.get("max_input_channels", 0)))

        if not candidates:
            for idx, d in enumerate(devices):
                if d.get("hostapi") != wasapi_index:
                    continue
                if d.get("max_input_channels", 0) <= 0:
                    continue
                name_l = d.get("name", "").lower()
                if "what u hear" in name_l or "stereo mix" in name_l:
                    candidates.append((5, idx, d.get("name", ""), d.get("max_input_channels", 0)))

        if not candidates:
            raise RuntimeError(
                "No WASAPI loopback-like input device found.\n"
                "Try enabling 'Stereo Mix' / 'What U Hear' in Windows Sound settings."
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0]
        return best[1], best[2], candidates

    @staticmethod
    def pick_supported_samplerate(dev_id: int, channels: int) -> int:
        candidates = [48000, 44100, 32000, 16000, 96000]
        for sr in candidates:
            try:
                sd.check_input_settings(device=dev_id, samplerate=sr, channels=channels, dtype="float32")
                return sr
            except Exception:
                pass

        d = sd.query_devices(dev_id)
        default_sr = int(d.get("default_samplerate", 44100))
        try:
            sd.check_input_settings(device=dev_id, samplerate=default_sr, channels=channels, dtype="float32")
            return default_sr
        except Exception as e:
            raise RuntimeError(
                f"No supported samplerate found for device {dev_id}. "
                f"default_samplerate={default_sr}. Last error={e}"
            )


# =======================
# Text post-process & delta logic
# =======================
class TextPostProcessor:
    def __init__(self):
        self._ws_re = re.compile(r"\s+")

    def normalize(self, s: str) -> str:
        return " ".join(s.strip().split())

    def similarity_ratio(self, a: str, b: str) -> float:
        a = self.normalize(a).lower()
        b = self.normalize(b).lower()
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        sa = set(a.split())
        sb = set(b.split())
        return len(sa & sb) / max(1, len(sa | sb))

    def compress_run_words(self, text: str, max_repeat: int = 1) -> str:
        w = text.split()
        if not w:
            return text
        out = []
        run_word = None
        run_len = 0
        for tok in w:
            t = tok.lower()
            if t == run_word:
                run_len += 1
                if run_len <= max_repeat:
                    out.append(tok)
            else:
                run_word = t
                run_len = 1
                out.append(tok)
        return " ".join(out)

    def dedup_tail_words(self, text: str, max_ngram: int = 6) -> str:
        words = text.split()
        for n in range(max_ngram, 1, -1):
            if len(words) >= 2 * n:
                a = [w.lower() for w in words[-2 * n:-n]]
                b = [w.lower() for w in words[-n:]]
                if a == b:
                    words = words[:-n]
                    return " ".join(words)
        return text

    def drop_looping_phrases(self, text: str, phrase_max_words: int = 4, repeats: int = 4) -> str:
        w = text.split()
        if len(w) < phrase_max_words * repeats:
            return text
        low = [x.lower() for x in w]
        for k in range(2, phrase_max_words + 1):
            phrase = low[:k]
            ok = True
            for r in range(1, repeats):
                if low[r * k:(r + 1) * k] != phrase:
                    ok = False
                    break
            if ok:
                cut = repeats * k
                return " ".join(w[cut:]).strip()
        return text

    def clean_out(self, out: str) -> str:
        out = self.normalize(out)
        out = self.drop_looping_phrases(out, phrase_max_words=4, repeats=4)
        out = self.compress_run_words(out, max_repeat=1)
        out = self.dedup_tail_words(out)
        return self.normalize(out)

    # ---- pending merge ----
    def append_pending(self, pending: str, chunk: str) -> str:
        pending = self.normalize(pending)
        chunk = self.normalize(chunk)
        if not chunk:
            return pending
        if not pending:
            return chunk

        p_low = pending.lower()
        c_low = chunk.lower()

        if c_low in p_low:
            return pending

        p_words = pending.split()
        c_words = chunk.split()
        p_low_words = [w.lower() for w in p_words]
        c_low_join = " " + " ".join(w.lower() for w in c_words) + " "

        # tail overwrite
        tail_max = min(14, len(p_words))
        best_tail = 0
        for k in range(6, tail_max + 1):
            tail = " " + " ".join(p_low_words[-k:]) + " "
            if tail in c_low_join:
                best_tail = k
        if best_tail > 0:
            merged = " ".join(p_words[:-best_tail] + c_words)
            merged = self.normalize(merged)
            merged = self.dedup_tail_words(merged)
            merged = self.compress_run_words(merged, 2)
            return merged

        # char overlap
        N = 180
        p_tail = p_low[-N:]
        c_head = c_low[:N]
        best = 0
        max_k = min(len(p_tail), len(c_head))
        for k in range(12, max_k + 1):
            if p_tail[-k:] == c_head[:k]:
                best = k
        if best > 0:
            overlap = len(p_tail[-best:])
            merged = pending + chunk[overlap:]
            merged = self.normalize(merged)
            merged = self.dedup_tail_words(merged)
            merged = self.compress_run_words(merged, 2)
            return merged

        # word overlap
        p_low_words2 = [w.lower() for w in p_words]
        c_low_words2 = [w.lower() for w in c_words]

        max_k = min(len(p_words), len(c_words), 25)
        best_k = 0
        for k in range(1, max_k + 1):
            if p_low_words2[-k:] == c_low_words2[:k]:
                best_k = k

        if best_k > 0:
            merged = " ".join(p_words + c_words[best_k:])
        else:
            if p_low_words2 and c_low_words2 and p_low_words2[-1] == c_low_words2[0]:
                merged = " ".join(p_words + c_words[1:])
            else:
                merged = pending + " " + chunk

        merged = self.normalize(merged)
        merged = self.dedup_tail_words(merged)
        merged = self.compress_run_words(merged, 2)
        return merged

    # ---- delta (real-time) ----
    def delta_append(self, prev: str, curr: str) -> str:
        prev = self.normalize(prev)
        curr = self.normalize(curr)
        if not curr:
            return ""
        if not prev:
            return curr

        prev_low = prev.lower()
        curr_low = curr.lower()

        if curr_low.startswith(prev_low):
            delta = self.normalize(curr[len(prev):])
        else:
            pw = prev.split()
            cw = curr.split()
            pwl = [w.lower() for w in pw]
            cwl = [w.lower() for w in cw]
            max_k = min(len(pw), len(cw), 25)
            best_k = 0
            for k in range(1, max_k + 1):
                if pwl[-k:] == cwl[:k]:
                    best_k = k
            if best_k > 0:
                delta = " ".join(cw[best_k:]).strip()
            else:
                delta = curr

        delta = self.normalize(delta)
        if not delta:
            return ""

        # filter micro rewrites for real-time
        if len(delta.split()) <= 2:
            return ""

        delta = self.compress_run_words(delta, max_repeat=1)
        delta = self.dedup_tail_words(delta)
        return self.normalize(delta)

    # ---- delta (boundary flush) ----
    def delta_append_boundary(self, prev: str, curr: str) -> str:
        prev = self.normalize(prev)
        curr = self.normalize(curr)
        if not curr:
            return ""
        if not prev:
            return curr

        prev_low = prev.lower()
        curr_low = curr.lower()

        if curr_low.startswith(prev_low):
            return self.normalize(curr[len(prev):])

        pw = prev.split()
        cw = curr.split()
        pwl = [w.lower() for w in pw]
        cwl = [w.lower() for w in cw]
        max_k = min(len(pw), len(cw), 25)
        best_k = 0
        for k in range(1, max_k + 1):
            if pwl[-k:] == cwl[:k]:
                best_k = k
        if best_k > 0:
            return " ".join(cw[best_k:]).strip()

        return curr


# =======================
# Device detection (GPU/CPU)
# =======================
def detect_device() -> Tuple[str, str, bool]:
    try:
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {name}")
            return "cuda", "float16", True
        else:
            print("⚠️ CUDA not available -> using CPU")
            return "cpu", "int8", False
    except Exception as e:
        print(f"⚠️ CUDA check failed -> using CPU. Reason: {e}")
        return "cpu", "int8", False


# =======================
# Realtime Transcriber
# =======================
class RealtimeTranscriber:
    
    def dump_buffer_with_flush(self) -> str:

        self._final_transcribe_into_pending()

     
        final_line = self.text.normalize(self.pending_line)
        if final_line:
            delta = self.text.delta_append_boundary(self.last_printed_line, final_line).strip()
            if delta:
                self.emit(delta)
            self.pending_line = ""
            self.last_printed_line = ""

  
        return self.buf.dump_and_clear()
    
    
    def __init__(self, cfg: Config):
        self.buf = TranscriptBuffer()
        start_listener(self.dump_buffer_with_flush)
        
        self.cfg = cfg
        self.text = TextPostProcessor()

        self.device, self.compute_type, self.cuda_ok = detect_device()

        self.model = WhisperModel(cfg.model_size, device=self.device, compute_type=self.compute_type)
        self.vad = webrtcvad.Vad(cfg.vad_mode)

        # runtime state
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.ring = np.zeros((0,), dtype=np.float32)
        self.vad_flags: List[int] = []

        self.total_samples_seen = 0
        self.last_emit_samples = 0
        self.last_emitted_t = 0.0
        self.last_commit_samples = 0
        self.silence_frames = 0

        # pending
        self.pending_line = ""
        self.last_printed_line = ""

        self.last_pending_norm = ""
        self.pending_stable_count = 0

        # filled later after opening stream
        self.sr_in = None
        self.channels = None
        self.frame_len_in = None
        self.frame_len_asr = int(cfg.sr_asr * cfg.frame_ms / 1000)
        self.hop_samples = int(cfg.sr_asr * cfg.hop_sec)
        self.window_samples = int(cfg.sr_asr * cfg.window_sec)
        self.ring_max = int(cfg.sr_asr * max(cfg.ring_min_sec, cfg.window_sec + cfg.ring_extra_sec))

        self.vad_check_frames = max(1, int((cfg.vad_check_sec * 1000) / cfg.frame_ms))

    def emit(self,delta: str):
        delta = delta.strip()
        if not delta:
            return
        print(delta, flush=True)
        self.buf.append(delta)

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        self.audio_q.put(indata.copy())


    def _final_transcribe_into_pending(self):
        """
        Force one last transcribe right now (ignoring hop schedule),
        to capture the tail words before boundary/dump.
        """
        if len(self.ring) < self.window_samples:
            return

        x = self.ring[-self.window_samples:]
        if len(x) < self.window_samples:
            x = np.pad(x, (self.window_samples - len(x), 0), mode="constant")
        if len(x) < int(self.cfg.sr_asr * 0.2):
            return

        x = np.ascontiguousarray(x, dtype=np.float32)

        segments, _info = self.model.transcribe(
            x,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        # window timing
        window_end_t = self.total_samples_seen / self.cfg.sr_asr
        window_start_t = window_end_t - self.cfg.window_sec

        # small backtrack so we don't miss tail due to guard
        backtrack = 0.25  # seconds
        old_last_emitted = self.last_emitted_t
        self.last_emitted_t = max(0.0, self.last_emitted_t - backtrack)

        new_chunks, max_end_t = self._collect_new_chunks(segments, window_start_t)

        # restore last_emitted_t baseline later
        if not new_chunks:
            self.last_emitted_t = old_last_emitted
            return

        out = self.text.normalize(" ".join(new_chunks))
        out = self.text.clean_out(out)
        if out:
            self.pending_line = self.text.append_pending(self.pending_line, out)

        # advance emitted time to avoid reprocessing forever
        self.last_emitted_t = max(old_last_emitted, max_end_t)



    def _boundary_break(self):
        self._final_transcribe_into_pending()
        final_line = self.text.normalize(self.pending_line)

        if final_line:
            delta = self.text.delta_append_boundary(self.last_printed_line, final_line).strip()
            if delta:
                self.emit(delta)

            if final_line.rstrip().endswith((".", "?", "!")):
                print("", flush=True)

            self.last_printed_line = ""

        # reset boundary state
        self.pending_line = ""
        self.silence_frames = 0
        self.last_commit_samples = self.total_samples_seen
        self.last_emitted_t = self.total_samples_seen / self.cfg.sr_asr

        self.last_pending_norm = ""
        self.pending_stable_count = 0

    def _update_vad(self, mono16: np.ndarray):
        # 20ms frames @16k
        n_full = len(mono16) // self.frame_len_asr
        for i in range(n_full):
            frame = mono16[i * self.frame_len_asr:(i + 1) * self.frame_len_asr]
            is_speech = self.vad.is_speech(float_to_pcm16_bytes(frame), self.cfg.sr_asr)
            self.vad_flags.append(1 if is_speech else 0)
        if len(self.vad_flags) > 300:
            self.vad_flags = self.vad_flags[-300:]

    def _speech_ratio(self) -> float:
        recent = self.vad_flags[-self.vad_check_frames:] if len(self.vad_flags) >= self.vad_check_frames else self.vad_flags
        return (sum(recent) / max(1, len(recent))) if recent else 0.0

    def _collect_new_chunks(self, segments, window_start_t: float) -> Tuple[List[str], float]:
        new_chunks = []
        max_end_t = self.last_emitted_t

        for seg in segments:
            seg_dur = float(seg.end) - float(seg.start)
            if seg_dur < self.cfg.min_seg_dur_sec:
                continue

            g_end = window_start_t + float(seg.end)
            if g_end <= self.last_emitted_t + self.cfg.emit_guard_sec:
                continue

            t = self.text.normalize(seg.text)
            if len(t) >= self.cfg.min_print_chars:
                new_chunks.append(t)
                if g_end > max_end_t:
                    max_end_t = g_end

        return new_chunks, max_end_t

    def run(self):
        # device selection
        dev_id, dev_name, _ = DeviceSelector.pick_wasapi_loopback_device()
        print("✅ Using loopback input device:")
        print(f"   device_id = {dev_id}")
        print(f"   name      = {dev_name}")

        # samplerate + channels
        channels = self.cfg.ch_try
        try:
            sr_in = DeviceSelector.pick_supported_samplerate(dev_id, channels)
        except Exception:
            channels = 1
            sr_in = DeviceSelector.pick_supported_samplerate(dev_id, channels)

        self.sr_in = sr_in
        self.channels = channels
        self.frame_len_in = int(sr_in * self.cfg.frame_ms / 1000)

        print(f"✅ Open stream with samplerate={sr_in}, channels={channels}")
        print(f"✅ Whisper backend: device={self.device}, compute={self.compute_type}")
        print("✅ Listening to SYSTEM AUDIO -> English text. Ctrl+C to stop.")

        try:
            with sd.InputStream(
                device=dev_id,
                samplerate=sr_in,
                channels=channels,
                dtype="float32",
                callback=self._callback,
                blocksize=self.frame_len_in,
            ):
                while True:
                    block = self.audio_q.get()
                    mono = stereo_to_mono(block)

                    mono16 = resample_linear(mono, sr_in, self.cfg.sr_asr)

                    self.ring = np.concatenate([self.ring, mono16])
                    if len(self.ring) > self.ring_max:
                        self.ring = self.ring[-self.ring_max:]

                    self.total_samples_seen += len(mono16)

                    self._update_vad(mono16)

                    # hop schedule
                    if self.total_samples_seen - self.last_emit_samples < self.hop_samples:
                        continue
                    self.last_emit_samples = self.total_samples_seen

                    if len(self.ring) < self.window_samples:
                        continue

                    # silence tracking
                    speech_ratio = self._speech_ratio()
                    if speech_ratio < self.cfg.display_silence_ratio:
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0

                    # timeout boundary
                    elapsed_sec = (self.total_samples_seen - self.last_commit_samples) / self.cfg.sr_asr
                    if self.pending_line and elapsed_sec >= self.cfg.max_sentence_sec:
                        self._boundary_break()
                        continue

                    # mostly silence: boundary by pause, skip transcribe
                    if speech_ratio < self.cfg.vad_speech_ratio:
                        if self.pending_line and self.silence_frames >= self.cfg.silence_break_frames:
                            self._boundary_break()
                        continue

                    # ASR window
                    x = self.ring[-self.window_samples:]
                    if len(x) < self.window_samples:
                        x = np.pad(x, (self.window_samples - len(x), 0), mode="constant")
                    if len(x) < int(self.cfg.sr_asr * 0.2):
                        continue

                    x = np.ascontiguousarray(x, dtype=np.float32)

                    segments, _info = self.model.transcribe(
                        x,
                        language=self.cfg.language,
                        beam_size=self.cfg.beam_size,
                        vad_filter=False,
                        temperature=0.0,
                        condition_on_previous_text=False,
                    )

                    window_end_t = self.total_samples_seen / self.cfg.sr_asr
                    window_start_t = window_end_t - self.cfg.window_sec

                    new_chunks, max_end_t = self._collect_new_chunks(segments, window_start_t)

                    if not new_chunks:
                        continue

                    out = self.text.normalize(" ".join(new_chunks))
                    out = self.text.clean_out(out)
                    if not out:
                        continue

                    # merge into pending
                    self.pending_line = self.text.append_pending(self.pending_line, out)
                    p_norm = self.text.normalize(self.pending_line)

                    # stable gating
                    if len(p_norm) >= self.cfg.pending_stable_min_chars:
                        sim = self.text.similarity_ratio(p_norm, self.last_pending_norm)
                        if sim >= self.cfg.pending_similarity:
                            self.pending_stable_count += 1
                        else:
                            self.pending_stable_count = 1
                            self.last_pending_norm = p_norm

                        if self.pending_stable_count >= self.cfg.pending_stable_required:
                            delta = self.text.delta_append(self.last_printed_line, p_norm).strip()
                            if delta:
                                self.emit(delta)
                                self.last_printed_line = p_norm
                            self.pending_stable_count = 0
                    else:
                        self.last_pending_norm = p_norm
                        self.pending_stable_count = 0

                    self.last_emitted_t = max_end_t

        except KeyboardInterrupt:
            final_line = self.text.normalize(self.pending_line)
            delta = self.text.delta_append_boundary(self.last_printed_line, final_line).strip()
            if delta:
                self.emit(delta)
            print("\nStopped.")


# =======================
# Entrypoint
# =======================
if __name__ == "__main__":
    cfg = Config()
    RealtimeTranscriber(cfg).run()