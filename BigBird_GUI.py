# BigBird_GUI_Qt.py
# PySide6 GUI for BigBird control server
# Requires: pip install PySide6 requests

from PySide6 import QtCore, QtGui, QtWidgets
import requests
import sys
import os
import shutil
import time
import math
import functools
import urllib.parse
from datetime import datetime

API_BASE = "http://127.0.0.1:8765"

DEFAULT_PYTHON = sys.executable  # change if you want a specific interpreter
DEFAULT_API_SCRIPT = os.path.expanduser("/Users/x/Projects/Kratt/BigBird/BigBirdApi.py")  # adjust if different

# ---------- Small HTTP client ----------

class ApiClient:
    def __init__(self, base=API_BASE, timeout=2.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def get(self, path):
        try:
            r = requests.get(self.base + path, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def post(self, path, payload):
        try:
            r = requests.post(self.base + path, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, path):
        try:
            r = requests.delete(self.base + path, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

api = ApiClient()

# ---------- Widgets ----------

def format_when(val):
    """Format created_at into DD/MM HH:MM (local time). Accepts epoch seconds/ms or ISO strings."""
    try:
        if val in (None, "", "—"):
            return "—"
        # Numeric epoch (seconds or ms)
        if isinstance(val, (int, float)):
            ts = float(val)
            if ts > 1e12:  # likely ms
                ts /= 1000.0
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%d/%m %H:%M")
        # String input
        s = str(val).strip()
        # Digits → epoch
        if s.isdigit():
            ts = float(s)
            if ts > 1e12:
                ts /= 1000.0
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%d/%m %H:%M")
        # ISO-like
        s2 = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s2)
        except Exception:
            # try without fractional seconds/timezone
            try:
                dt = datetime.fromisoformat(s2.split(".")[0])
            except Exception:
                return s  # give up gracefully
        # If aware, convert to local
        try:
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.astimezone()
        except Exception:
            pass
        return dt.strftime("%d/%m %H:%M")
    except Exception:
        return str(val)

def add_row(form: QtWidgets.QFormLayout, label: str, widget: QtWidgets.QWidget):
    lab = QtWidgets.QLabel(label)
    form.addRow(lab, widget)
    return widget


# set_if_idle is now replaced with MainWindow._set_if_idle for all uses in MainWindow.

class ChatView(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.setFont(mono)

    def render_messages(self, msgs):
        # msgs: [{ts, role, text}]
        self.setPlainText("\n".join(f"{m.get('role','?'):>9}: {m.get('text','')}" for m in msgs))

# ---------- Main Window ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BigBird Control (PySide6)")
        self.resize(1100, 780)

        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        # ---- Top bar ----
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        # ---- API launcher row ----
        api_row = QtWidgets.QHBoxLayout()
        root.addLayout(api_row)

        api_row.addWidget(QtWidgets.QLabel("Python:"))
        self.py_path = QtWidgets.QLineEdit(DEFAULT_PYTHON)
        self.py_path.setMinimumWidth(260)
        api_row.addWidget(self.py_path)

        api_row.addWidget(QtWidgets.QLabel("BigBirdApi.py:"))
        self.api_path = QtWidgets.QLineEdit(DEFAULT_API_SCRIPT)
        self.api_path.setMinimumWidth(360)
        api_row.addWidget(self.api_path)

        self.api_browse = QtWidgets.QPushButton("Browse…")
        api_row.addWidget(self.api_browse)

        self.start_api_btn = QtWidgets.QPushButton("Start BigBirdApi")
        api_row.addWidget(self.start_api_btn)

        self.stop_api_btn = QtWidgets.QPushButton("Stop BigBirdApi")
        self.stop_api_btn.setEnabled(False)
        api_row.addWidget(self.stop_api_btn)

        # Engine
        top.addWidget(QtWidgets.QLabel("Engine:"))
        self.engine = QtWidgets.QComboBox()
        self.engine.addItems(["playht", "elevenlabs"])
        top.addWidget(self.engine)
        self.apply_engine_btn = QtWidgets.QPushButton("Apply")
        top.addWidget(self.apply_engine_btn)

        # Streaming
        self.streaming = QtWidgets.QCheckBox("Streaming")
        top.addWidget(self.streaming)

        # Audio output
        top.addWidget(QtWidgets.QLabel("Output:"))
        self.output = QtWidgets.QComboBox()
        self.output.addItems(["default", "blackhole"])
        top.addWidget(self.output)
        self.apply_audio_btn = QtWidgets.QPushButton("Apply Audio")
        top.addWidget(self.apply_audio_btn)

        top.addStretch(1)

        # System info (cpu/mem/clones)
        self.sys_label = QtWidgets.QLabel("cpu: —   mem: —   clones: —")
        self.sys_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.sys_label)

        # ---- Middle area: chat (left) + tabs (right) ----
        mid = QtWidgets.QHBoxLayout()
        root.addLayout(mid, 1)

        # Chat
        left = QtWidgets.QVBoxLayout()
        mid.addLayout(left, 2)
        self.chat = ChatView()
        left.addWidget(self.chat, 1)

        # Speak box
        speak_row = QtWidgets.QHBoxLayout()
        left.addLayout(speak_row)
        self.say_entry = QtWidgets.QLineEdit()
        self.say_entry.setPlaceholderText("Type something to speak via current TTS…")
        speak_row.addWidget(self.say_entry, 1)
        self.say_btn = QtWidgets.QPushButton("Speak")
        speak_row.addWidget(self.say_btn)

        # Controls (tabs)
        tabs = QtWidgets.QTabWidget()
        mid.addWidget(tabs, 1)

        # --- Presets tab ---
        self.presets_tab = QtWidgets.QWidget()
        tabs.addTab(self.presets_tab, "Presets")
        pr = QtWidgets.QVBoxLayout(self.presets_tab)

        self.presets_table = QtWidgets.QTableWidget(0, 2)
        self.presets_table.setHorizontalHeaderLabels(["★", "Name"])
        self.presets_table.horizontalHeader().setStretchLastSection(True)
        self.presets_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.presets_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        pr.addWidget(self.presets_table, 1)

        prow = QtWidgets.QHBoxLayout()
        self.preset_new_btn = QtWidgets.QPushButton("Save Current…")
        self.preset_apply_btn = QtWidgets.QPushButton("Apply")
        self.preset_rename_btn = QtWidgets.QPushButton("Rename…")
        self.preset_delete_btn = QtWidgets.QPushButton("Delete")
        self.preset_star_btn = QtWidgets.QPushButton("Star")
        self.preset_up_btn = QtWidgets.QPushButton("Move Up")
        self.preset_down_btn = QtWidgets.QPushButton("Move Down")
        for b in (self.preset_new_btn, self.preset_apply_btn, self.preset_rename_btn, self.preset_delete_btn, self.preset_star_btn, self.preset_up_btn, self.preset_down_btn):
            prow.addWidget(b)
        prow.addStretch(1)
        pr.addLayout(prow)

        # --- PlayHT tab ---
        self.playht_tab = QtWidgets.QWidget()
        tabs.addTab(self.playht_tab, "PlayHT")
        pform = QtWidgets.QFormLayout(self.playht_tab)

        self.p_speed = add_row(pform, "Speed",
                               self._dbl(0.0, 3.0, 0.01, 1.0))
        self.p_style = add_row(pform, "Style guidance",
                               self._dbl(0.0, 10.0, 0.1, 1.0))
        self.p_voice = add_row(pform, "Voice guidance",
                               self._dbl(0.0, 10.0, 0.1, 1.0))
        self.p_temp  = add_row(pform, "Temperature",
                               self._dbl(0.0, 3.0, 0.01, 1.5))
        self.p_textg = add_row(pform, "Text guidance",
                               self._dbl(0.0, 10.0, 0.1, 0.0))
        self.p_rate  = add_row(pform, "Sample rate (Hz)",
                               self._int(16000, 48000, 1000, 24000))

        self.apply_playht = QtWidgets.QPushButton("Apply PlayHT")
        pform.addRow(self.apply_playht)

        # --- ElevenLabs tab ---
        self.eleven_tab = QtWidgets.QWidget()
        tabs.addTab(self.eleven_tab, "ElevenLabs")
        eform = QtWidgets.QFormLayout(self.eleven_tab)

        self.e_boost = QtWidgets.QCheckBox("Enable speaker boost")
        eform.addRow(self.e_boost)
        self.e_stab  = add_row(eform, "Stability",        self._dbl(0.0, 1.5, 0.01, 0.5))
        self.e_sim   = add_row(eform, "Similarity boost", self._dbl(0.0, 1.5, 0.01, 0.8))
        self.e_style = add_row(eform, "Style",            self._dbl(0.0, 1.5, 0.01, 0.1))
        self.e_speed = add_row(eform, "Speed",            self._dbl(0.5, 2.0, 0.01, 1.0))

        self.apply_eleven = QtWidgets.QPushButton("Apply ElevenLabs")
        eform.addRow(self.apply_eleven)

        # --- VAD tab ---
        self.vad_tab = QtWidgets.QWidget()
        tabs.addTab(self.vad_tab, "VAD")
        vform = QtWidgets.QFormLayout(self.vad_tab)

        self.v_mode = add_row(vform, "Aggressiveness (0–3)",
                              self._int(0, 3, 1, 2))
        self.v_init = add_row(vform, "Initial silence timeout (s)",
                              self._dbl(0.0, 60.0, 0.5, 15.0))
        self.v_sil  = add_row(vform, "Max trailing silence (s)",
                              self._dbl(0.0, 5.0, 0.1, 1.0))

        self.apply_vad = QtWidgets.QPushButton("Apply VAD")
        vform.addRow(self.apply_vad)

        # --- Arduino tab ---
        self.arduino_tab = QtWidgets.QWidget()
        tabs.addTab(self.arduino_tab, "Arduino")
        aform = QtWidgets.QFormLayout(self.arduino_tab)
        self.ard_status = QtWidgets.QLabel("port: —   connected: —   override(hook/led): —/—")
        aform.addRow(self.ard_status)

        # Hook override (latched buttons)
        hook_row = QtWidgets.QHBoxLayout()
        hook_row.addWidget(QtWidgets.QLabel("Hook override:"))

        self.hook_auto_btn = QtWidgets.QToolButton()
        self.hook_auto_btn.setText("Auto")
        self.hook_auto_btn.setCheckable(True)
        self.hook_auto_btn.setAutoExclusive(True)

        self.hook_off_btn = QtWidgets.QToolButton()
        self.hook_off_btn.setText("Off-hook")
        self.hook_off_btn.setCheckable(True)
        self.hook_off_btn.setAutoExclusive(True)

        self.hook_on_btn = QtWidgets.QToolButton()
        self.hook_on_btn.setText("On-hook")
        self.hook_on_btn.setCheckable(True)
        self.hook_on_btn.setAutoExclusive(True)

        # default latched state
        self.hook_auto_btn.setChecked(True)

        # wire clicks (only send when becoming True)
        self.hook_auto_btn.clicked.connect(lambda checked: checked and self.on_hook_latch("auto"))
        self.hook_off_btn.clicked.connect(lambda checked: checked and self.on_hook_latch("offhook"))
        self.hook_on_btn.clicked.connect(lambda checked: checked and self.on_hook_latch("onhook"))

        hook_row.addWidget(self.hook_auto_btn)
        hook_row.addWidget(self.hook_off_btn)
        hook_row.addWidget(self.hook_on_btn)
        hook_row.addStretch(1)
        aform.addRow(hook_row)

        # LED controls
        led_row = QtWidgets.QHBoxLayout()
        led_row.addWidget(QtWidgets.QLabel("LED:"))
        for label in ("OFF", "RECORDING", "REPLYING", "PROCESSING"):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(functools.partial(self.on_led_set, label))
            led_row.addWidget(btn)
        self.led_clear_btn = QtWidgets.QPushButton("Clear LED Override")
        self.led_clear_btn.clicked.connect(lambda: self.on_led_set(None))
        led_row.addWidget(self.led_clear_btn)
        aform.addRow(led_row)

        # Ring controls
        ring_row = QtWidgets.QHBoxLayout()
        ring_row.addWidget(QtWidgets.QLabel("Ring:"))
        self.ring_single_btn = QtWidgets.QPushButton("Single")
        self.ring_single_btn.clicked.connect(lambda: self.on_ring("single"))
        ring_row.addWidget(self.ring_single_btn)
        self.ring_double_btn = QtWidgets.QPushButton("Double")
        self.ring_double_btn.clicked.connect(lambda: self.on_ring("double"))
        ring_row.addWidget(self.ring_double_btn)
        aform.addRow(ring_row)

        # --- Cloning tab ---
        self.cloning_tab = QtWidgets.QWidget()
        tabs.addTab(self.cloning_tab, "Cloning")
        clform = QtWidgets.QVBoxLayout(self.cloning_tab)

        # Progress bar and info line
        self.clone_bar = QtWidgets.QProgressBar()
        self.clone_bar.setRange(0, 100)
        self.clone_bar.setValue(0)
        clform.addWidget(self.clone_bar)

        info_row = QtWidgets.QHBoxLayout()
        self.clone_info = QtWidgets.QLabel("0.0/0.0s (rem 0.0s)")
        info_row.addWidget(self.clone_info)
        self.clone_counter = QtWidgets.QLabel("clones this session: 0")
        info_row.addWidget(self.clone_counter)
        self.clone_pending = QtWidgets.QLabel("pending clips: 0")
        info_row.addWidget(self.clone_pending)
        info_row.addStretch(1)
        clform.addLayout(info_row)

        # Threshold and Clone Now controls
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Required seconds:"))
        self.clone_req = QtWidgets.QDoubleSpinBox()
        self.clone_req.setRange(1.0, 30.0)
        self.clone_req.setSingleStep(0.5)
        self.clone_req.setDecimals(1)
        self.clone_req.setValue(7.5)
        row2.addWidget(self.clone_req)
        self.clone_req_apply = QtWidgets.QPushButton("Apply")
        row2.addWidget(self.clone_req_apply)
        row2.addSpacing(20)
        row2.addWidget(QtWidgets.QLabel("Engine:"))
        self.clone_engine = QtWidgets.QComboBox()
        self.clone_engine.addItems(["playht", "elevenlabs"])
        row2.addWidget(self.clone_engine)
        row2.addWidget(QtWidgets.QLabel("Voice name:"))
        self.clone_name = QtWidgets.QLineEdit()
        self.clone_name.setPlaceholderText("session_cloneYYYYMMDD-HHMMSS")
        row2.addWidget(self.clone_name, 1)
        self.clone_now_btn = QtWidgets.QPushButton("Clone Now")
        row2.addWidget(self.clone_now_btn)
        clform.addLayout(row2)

        # Identity info
        self.identity_info = QtWidgets.QPlainTextEdit()
        self.identity_info.setReadOnly(True)
        self.identity_info.setPlaceholderText("savedInfo.txt excerpt will appear here…")
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.identity_info.setFont(mono)
        self.identity_info.setMaximumHeight(140)
        clform.addWidget(self.identity_info)

        # --- Clones tab ---
        self.clones_tab = QtWidgets.QWidget()
        tabs.addTab(self.clones_tab, "Clones")
        cbox = QtWidgets.QVBoxLayout(self.clones_tab)

        # Table
        self.clones_table = QtWidgets.QTableWidget(0, 5)
        self.clones_table.setHorizontalHeaderLabels(["Name", "ID", "Engine", "Created", "Actions"])
        self.clones_table.horizontalHeader().setStretchLastSection(True)
        self.clones_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.clones_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        cbox.addWidget(self.clones_table, 1)

        # Footer controls
        cfooter = QtWidgets.QHBoxLayout()
        self.clones_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.clones_refresh_btn.clicked.connect(self.refresh_clones_now)
        cfooter.addWidget(self.clones_refresh_btn)

        self.clones_sync_btn = QtWidgets.QPushButton("Sync")
        self.clones_sync_btn.clicked.connect(self.on_clones_sync)
        cfooter.addWidget(self.clones_sync_btn)

        cfooter.addWidget(QtWidgets.QLabel("Evict oldest:"))
        self.clones_engine = QtWidgets.QComboBox()
        self.clones_engine.addItems(["All", "playht", "elevenlabs"])
        cfooter.addWidget(self.clones_engine)
        self.clones_evict_btn = QtWidgets.QPushButton("Evict")
        self.clones_evict_btn.clicked.connect(self.on_clones_evict)
        cfooter.addWidget(self.clones_evict_btn)

        cfooter.addStretch(1)
        cbox.addLayout(cfooter)

        # --- Logs tab ---
        self.logs_tab = QtWidgets.QWidget()
        tabs.addTab(self.logs_tab, "Logs")
        lform = QtWidgets.QVBoxLayout(self.logs_tab)
        self.logs = QtWidgets.QPlainTextEdit()
        self.logs.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.logs.setFont(mono)
        lform.addWidget(self.logs, 1)
        self.clear_logs_btn = QtWidgets.QPushButton("Clear logs")
        lform.addWidget(self.clear_logs_btn, 0, QtCore.Qt.AlignRight)

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Wire up actions
        self.apply_engine_btn.clicked.connect(self.on_apply_engine)
        self.apply_audio_btn.clicked.connect(self.on_apply_audio)
        self.apply_playht.clicked.connect(self.on_apply_playht)
        self.apply_eleven.clicked.connect(self.on_apply_eleven)
        self.apply_vad.clicked.connect(self.on_apply_vad)
        self.say_btn.clicked.connect(self.on_say)

        self.start_api_btn.clicked.connect(self.on_start_api)
        self.stop_api_btn.clicked.connect(self.on_stop_api)
        self.api_browse.clicked.connect(self.on_browse_api)
        self.clear_logs_btn.clicked.connect(self.logs.clear)
        self.clone_req_apply.clicked.connect(self.on_clone_threshold)
        self.clone_now_btn.clicked.connect(self.on_clone_now)

        # Preset actions
        self.preset_new_btn.clicked.connect(self.on_preset_new)
        self.preset_apply_btn.clicked.connect(self.on_preset_apply)
        self.preset_rename_btn.clicked.connect(self.on_preset_rename)
        self.preset_delete_btn.clicked.connect(self.on_preset_delete)
        self.preset_star_btn.clicked.connect(self.on_preset_star)
        self.preset_up_btn.clicked.connect(lambda: self.on_preset_move(-1))
        self.preset_down_btn.clicked.connect(lambda: self.on_preset_move(1))

        # Subprocess (BigBirdApi) management
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_proc_output)
        self.proc.readyReadStandardError.connect(self.on_proc_output)
        self.proc.stateChanged.connect(self.on_proc_state)

        # Mark controls dirty on user interaction (prevents flick-back during edits)
        self.engine.currentIndexChanged.connect(lambda _: self._dirty(self.engine))
        self.output.currentIndexChanged.connect(lambda _: self._dirty(self.output))
        self.streaming.stateChanged.connect(lambda _: self._dirty(self.streaming))
        for w in (self.p_speed, self.p_style, self.p_voice, self.p_temp, self.p_textg, self.p_rate):
            w.valueChanged.connect(lambda _=None, ww=w: self._dirty(ww))
        for w in (self.e_stab, self.e_sim, self.e_style, self.e_speed):
            w.valueChanged.connect(lambda _=None, ww=w: self._dirty(ww))
        self.e_boost.stateChanged.connect(lambda _: self._dirty(self.e_boost))
        for w in (self.v_mode, self.v_init, self.v_sil):
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(lambda _=None, ww=w: self._dirty(ww))
        self.clone_req.valueChanged.connect(lambda _=None, ww=self.clone_req: self._dirty(ww))

        # Poll timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_state)
        self.timer.start(800)  # ms

        # First refresh
        self.refresh_state()

        self._last_arduino = 0.0
        self._last_clones = 0.0
        self._last_presets = 0.0


    def on_hook_latch(self, which):
        # which: "auto" | "offhook" | "onhook"
        payload = {"state": None if which == "auto" else which}
        res = api.post("/arduino/hook", payload)
        ok = res.get("ok", False)
        self.status.showMessage("Hook set" if ok else f"Hook error: {res.get('error','')}", 2000)

    # ---- "Dirty" state helpers ----
    def _dirty(self, widget, sec: float = 2.0):
        if not hasattr(self, "_dirty_until"):
            self._dirty_until = {}
        self._dirty_until[widget] = time.monotonic() + sec

    def _clear_dirty(self, *widgets):
        if not hasattr(self, "_dirty_until"):
            self._dirty_until = {}
        for w in widgets:
            self._dirty_until[w] = 0

    def _is_dirty(self, widget):
        if not hasattr(self, "_dirty_until"):
            self._dirty_until = {}
        return time.monotonic() < self._dirty_until.get(widget, 0)

    def _set_if_idle(self, widget, set_fn, value):
        # Skip remote overwrite if user is interacting or widget is marked dirty
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QLineEdit, QtWidgets.QCheckBox)):
            if widget.hasFocus() or self._is_dirty(widget):
                return
        if isinstance(widget, QtWidgets.QComboBox):
            try:
                if widget.view().isVisible():
                    return
            except Exception:
                pass
            if widget.hasFocus() or self._is_dirty(widget):
                return
        set_fn(value)

    # ---- helpers for spinboxes ----
    def _dbl(self, lo, hi, step, val):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(3 if step < 0.1 else 2)
        w.setValue(val)
        w.setAlignment(QtCore.Qt.AlignRight)
        return w

    def _int(self, lo, hi, step, val):
        w = QtWidgets.QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(val)
        w.setAlignment(QtCore.Qt.AlignRight)
        return w

    # ---- Apply handlers ----
    def on_apply_engine(self):
        e = self.engine.currentText().strip().lower()
        res = api.post("/engine", {"engine": e})
        self.status.showMessage(f"Engine → {e} ({'ok' if res.get('ok', True) else res.get('error','err')})", 2000)
        self._clear_dirty(self.engine)

    def on_apply_audio(self):
        payload = {
            "audio_output": self.output.currentText().strip().lower(),
            "streaming": bool(self.streaming.isChecked()),
        }
        res = api.post("/audio", payload)
        self.status.showMessage(f"Audio updated ({'ok' if res.get('ok', True) else res.get('error','err')})", 2000)
        self._clear_dirty(self.output, self.streaming)

    def on_apply_playht(self):
        payload = {
            "speed": float(self.p_speed.value()),
            "style_guidance": float(self.p_style.value()),
            "voice_guidance": float(self.p_voice.value()),
            "temperature": float(self.p_temp.value()),
            "text_guidance": float(self.p_textg.value()),
            "sample_rate": int(self.p_rate.value()),
        }
        res = api.post("/playht", payload)
        self.status.showMessage(f"PlayHT updated ({'ok' if res.get('ok', True) else res.get('error','err')})", 2000)
        self._clear_dirty(self.p_speed, self.p_style, self.p_voice, self.p_temp, self.p_textg, self.p_rate)

    def on_apply_eleven(self):
        payload = {
            "use_speaker_boost": bool(self.e_boost.isChecked()),
            "stability": float(self.e_stab.value()),
            "similarity_boost": float(self.e_sim.value()),
            "style": float(self.e_style.value()),
            "speed": float(self.e_speed.value()),
        }
        res = api.post("/elevenlabs", payload)
        self.status.showMessage(f"ElevenLabs updated ({'ok' if res.get('ok', True) else res.get('error','err')})", 2000)
        self._clear_dirty(self.e_boost, self.e_stab, self.e_sim, self.e_style, self.e_speed)

    def on_apply_vad(self):
        payload = {
            "mode": int(self.v_mode.value()),
            "initial_silence_timeout": float(self.v_init.value()),
            "max_silence_length": float(self.v_sil.value()),
        }
        res = api.post("/vad", payload)
        self.status.showMessage(f"VAD updated ({'ok' if res.get('ok', True) else res.get('error','err')})", 2000)
        self._clear_dirty(self.v_mode, self.v_init, self.v_sil)

    def on_say(self):
        txt = self.say_entry.text().strip()
        if not txt:
            return
        res = api.post("/speak", {"text": txt})
        ok = res.get("ok", False)
        self.status.showMessage("Speaking…" if ok else f"Speak error: {res.get('error','unknown')}", 2000)
        if ok:
            self.say_entry.clear()

    # ---- Arduino actions ----
    # (on_apply_hook removed as no longer used)
    def on_clone_threshold(self):
        secs = float(self.clone_req.value())
        res = api.post("/cloning/threshold", {"seconds": secs})
        ok = res.get("ok", False)
        self.status.showMessage("Threshold updated" if ok else f"Threshold error: {res.get('error','')}", 2000)
        if ok:
            self._clear_dirty(self.clone_req)

    def on_clone_now(self):
        eng = self.clone_engine.currentText().strip().lower()
        name = self.clone_name.text().strip()
        if not name:
            name = time.strftime("session_clone%Y%m%d-%H%M%S")
        res = api.post("/cloning/trigger", {"engine": eng, "voice_name": name})
        ok = res.get("ok", False)
        self.status.showMessage("Clone triggered" if ok else f"Clone error: {res.get('error','')}", 2000)

    def on_led_set(self, state):
        # state can be None to clear
        res = api.post("/arduino/led", {"state": state})
        ok = res.get("ok", False)
        msg = "LED updated" if ok else f"LED error: {res.get('error','')}"
        self.status.showMessage(msg, 2000)

    def on_ring(self, pattern):
        res = api.post("/arduino/ring", {"pattern": pattern})
        ok = res.get("ok", False)
        self.status.showMessage("Ringing…" if ok else f"Ring error: {res.get('error','')}", 2000)

    def refresh_arduino(self, force=False):
        now = time.monotonic()
        if not force and (now - getattr(self, "_last_arduino", 0.0) < 1.2):
            return
        st = api.get("/arduino/state") or {}
        self._last_arduino = now
        port = st.get("port", "—")
        conn = st.get("connected", False)
        ov = st.get("override", {})
        hook = ov.get("hook", "auto")
        led = ov.get("led", "auto")
        self.ard_status.setText(f"port: {port}   connected: {conn}   override(hook/led): {hook}/{led}")
        val = "auto" if hook in (None, "auto") else str(hook)
        for btn, name in (
            (self.hook_auto_btn, "auto"),
            (self.hook_off_btn, "offhook"),
            (self.hook_on_btn,  "onhook"),
        ):
            old = btn.blockSignals(True)
            btn.setChecked(val == name)
            btn.blockSignals(old)

    # ---- Clones actions ----
    def populate_clones(self, clones):
        self.clones_table.setRowCount(0)
        for c in clones:
            row = self.clones_table.rowCount()
            self.clones_table.insertRow(row)
            name = c.get("name") or c.get("title") or "—"
            vid = c.get("id") or "—"
            eng = (c.get("engine") or c.get("provider") or "—").lower()
            category = (c.get("category") or "").lower()
            is_owner = c.get("is_owner", None)
            raw_created = c.get("created_at") or c.get("created") or "—"
            formatted_created = format_when(raw_created)

            # Decide deletable status (fallback if API field missing)
            deletable = c.get("deletable", None)
            if deletable is None:
                if eng == "elevenlabs":
                    deletable = bool(category in ("cloned", "generated") and (is_owner in (True, None)))
                else:
                    deletable = True
            else:
                deletable = bool(deletable)

            eng_display = f"{eng} ({category})" if category else eng

            # cells
            it_name = QtWidgets.QTableWidgetItem(str(name))
            it_id = QtWidgets.QTableWidgetItem(str(vid))
            it_eng = QtWidgets.QTableWidgetItem(str(eng_display))
            it_cr = QtWidgets.QTableWidgetItem(str(formatted_created))
            it_cr.setToolTip(str(raw_created))
            # stash id on the first item for convenience
            it_name.setData(QtCore.Qt.UserRole, vid)
            self.clones_table.setItem(row, 0, it_name)
            self.clones_table.setItem(row, 1, it_id)
            self.clones_table.setItem(row, 2, it_eng)
            self.clones_table.setItem(row, 3, it_cr)

            # actions
            cell = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(cell)
            h.setContentsMargins(0,0,0,0)
            btn_del = QtWidgets.QPushButton("Delete")
            if not deletable:
                btn_del.setEnabled(False)
                tip = "Not deletable"
                if eng == "elevenlabs":
                    if category and category not in ("cloned", "generated"):
                        tip = "Cannot delete ElevenLabs premade/library voice"
                    elif is_owner is False:
                        tip = "Cannot delete (not owned in ElevenLabs)"
                btn_del.setToolTip(tip)
            else:
                btn_del.clicked.connect(functools.partial(self.on_clone_delete, vid))
            h.addWidget(btn_del)
            h.addStretch(1)
            self.clones_table.setCellWidget(row, 4, cell)

    def on_clone_delete(self, voice_id):
        if not voice_id:
            return
        vid_enc = urllib.parse.quote(voice_id, safe="")
        res = api.delete(f"/clones/{vid_enc}")
        ok = res.get("ok", False)
        self.status.showMessage("Clone deleted" if ok else f"Delete error: {res.get('error','')}", 2000)
        self.refresh_clones_now()

    def on_clones_sync(self):
        res = api.post("/clones/sync", {})
        ok = res.get("ok", False)
        self.status.showMessage("Clones synced" if ok else f"Sync error: {res.get('error','')}", 2000)
        self.refresh_clones_now()

    def on_clones_evict(self):
        engine = self.clones_engine.currentText()
        payload = {}
        if engine.lower() in ("playht", "elevenlabs"):
            payload["engine"] = engine.lower()
        res = api.post("/clones/evict-oldest", payload)
        ok = res.get("ok", False)
        self.status.showMessage("Evicted oldest" if ok else f"Evict error: {res.get('error','')}", 2000)
        self.refresh_clones_now()

    def refresh_clones_now(self):
        # force an immediate re-populate on next tick
        self._last_clones = 0.0

    # ---- BigBirdApi process management ----
    def on_browse_api(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select BigBirdApi.py", os.path.dirname(self.api_path.text() or ""), "Python Files (*.py);;All Files (*)")
        if path:
            self.api_path.setText(path)

    def on_start_api(self):
        if self.proc.state() != QtCore.QProcess.NotRunning:
            self.status.showMessage("BigBirdApi already running.", 2000)
            return
        py = self.py_path.text().strip() or DEFAULT_PYTHON
        script = self.api_path.text().strip() or DEFAULT_API_SCRIPT
        if not os.path.isfile(script):
            QtWidgets.QMessageBox.warning(self, "File not found", f"Could not find script:\n{script}")
            return
        if not os.path.isfile(py) and not shutil.which(py):
            QtWidgets.QMessageBox.warning(self, "Python not found", f"Interpreter not found:\n{py}")
            return

        self.logs.appendPlainText(f"[gui] launching: {py} {script}\n")
        self.proc.setProgram(py)
        self.proc.setArguments([script])
        self.proc.setWorkingDirectory(os.path.dirname(script))
        self.proc.start()
        if not self.proc.waitForStarted(3000):
            QtWidgets.QMessageBox.critical(self, "Launch failed", "Failed to start BigBirdApi process.")
            return
        self.start_api_btn.setEnabled(False)
        self.stop_api_btn.setEnabled(True)
        self.status.showMessage("BigBirdApi launched.", 2000)

    def on_stop_api(self):
        if self.proc.state() == QtCore.QProcess.NotRunning:
            self.status.showMessage("BigBirdApi is not running.", 2000)
            return
        self.status.showMessage("Stopping BigBirdApi…", 2000)
        self.proc.terminate()
        if not self.proc.waitForFinished(3000):
            self.proc.kill()
            self.proc.waitForFinished(2000)
        self.start_api_btn.setEnabled(True)
        self.stop_api_btn.setEnabled(False)
        self.status.showMessage("BigBirdApi stopped.", 2000)

    @QtCore.Slot()
    def on_proc_output(self):
        data = self.proc.readAllStandardOutput()
        try:
            text = bytes(data).decode("utf-8", "ignore")
        except Exception:
            text = str(data)
        if text:
            self.logs.appendPlainText(text.rstrip())

    @QtCore.Slot()
    def on_proc_state(self, state):
        states = {
            QtCore.QProcess.NotRunning: "stopped",
            QtCore.QProcess.Starting: "starting",
            QtCore.QProcess.Running: "running",
        }
        st = states.get(state, str(state))
        self.status.showMessage(f"BigBirdApi: {st}", 1500)
        if state == QtCore.QProcess.NotRunning:
            self.start_api_btn.setEnabled(True)
            self.stop_api_btn.setEnabled(False)

    # ---- Poll state and mirror into UI ----
    def refresh_state(self):
        data = api.get("/state")
        if not data:
            self.status.showMessage("Disconnected from BigBird control server", 1000)
            return

        # top-line
        self._set_if_idle(self.engine, self.engine.setCurrentText, data.get("engine", "playht"))
        self._set_if_idle(self.output, self.output.setCurrentText, data.get("audio_output", "default"))
        self._set_if_idle(self.streaming, self.streaming.setChecked, bool(data.get("streaming", True)))

        sys = data.get("system", {})
        cpu = sys.get("cpu_percent", "—")
        mem = sys.get("mem_mb", "—")
        if isinstance(mem, (int, float)):
            mem = f"{int(round(mem))} MB"
        clones = sys.get("clones_count", "—")
        split = sys.get("clones_split", {}) or {}
        ph = split.get("playht")
        el = split.get("elevenlabs")
        split_str = f" (PH:{ph or 0} EL:{el or 0})" if (isinstance(ph, int) or isinstance(el, int)) else ""
        self.sys_label.setText(f"cpu: {cpu}%   mem: {mem}   clones: {clones}{split_str}")

        # chat
        self.chat.render_messages(data.get("messages", []))

        # playht
        ph = data.get("playht", {})
        self._set_if_idle(self.p_speed, self.p_speed.setValue, float(ph.get("speed", 1.0)))
        self._set_if_idle(self.p_style, self.p_style.setValue, float(ph.get("style_guidance", 1.0)))
        self._set_if_idle(self.p_voice, self.p_voice.setValue, float(ph.get("voice_guidance", 1.0)))
        self._set_if_idle(self.p_temp,  self.p_temp.setValue,  float(ph.get("temperature", 1.5)))
        self._set_if_idle(self.p_textg, self.p_textg.setValue, float(ph.get("text_guidance", 0.0)))
        self._set_if_idle(self.p_rate,  self.p_rate.setValue,  int(ph.get("sample_rate", 24000)))

        # eleven
        el = data.get("elevenlabs", {})
        self._set_if_idle(self.e_boost, self.e_boost.setChecked, bool(el.get("use_speaker_boost", False)))
        self._set_if_idle(self.e_stab,  self.e_stab.setValue,  float(el.get("stability", 0.5)))
        self._set_if_idle(self.e_sim,   self.e_sim.setValue,   float(el.get("similarity_boost", 0.8)))
        self._set_if_idle(self.e_style, self.e_style.setValue, float(el.get("style", 0.1)))
        self._set_if_idle(self.e_speed, self.e_speed.setValue, float(el.get("speed", 1.0)))

        # vad
        vad = data.get("vad", {})
        self._set_if_idle(self.v_mode, self.v_mode.setValue, int(vad.get("mode", 2)))
        self._set_if_idle(self.v_init, self.v_init.setValue, float(vad.get("initial_silence_timeout", 15.0)))
        self._set_if_idle(self.v_sil,  self.v_sil.setValue,  float(vad.get("max_silence_length", 1.0)))

        # cloning progress
        cloning = data.get("cloning", {}) or {}
        req = float(cloning.get("required_seconds", 0.0) or 0.0)
        col = float(cloning.get("collected_seconds", 0.0) or 0.0)
        pct = float(cloning.get("percent", 0.0) or 0.0)
        rem = float(cloning.get("seconds_remaining", max(0.0, req - col)))
        sess = int(cloning.get("clones_in_session", 0) or 0)
        pend = int(cloning.get("pending_clips", 0) or 0)
        ident = cloning.get("identity_info") or ""

        # UI updates
        self.clone_bar.setValue(int(max(0, min(100, round(pct * 100)))))
        self.clone_info.setText(f"{col:.1f}/{req:.1f}s (rem {rem:.1f}s)")
        self.clone_counter.setText(f"clones this session: {sess}")
        self.clone_pending.setText(f"pending clips: {pend}")
        # threshold spinbox: respect dirty state
        self._set_if_idle(self.clone_req, self.clone_req.setValue, req if req > 0 else self.clone_req.value())
        # identity text
        if isinstance(ident, str):
            self.identity_info.setPlainText(ident)

        # arduino status (throttled)
        self.refresh_arduino()

        # clones table (throttled every ~3s)
        now = time.monotonic()
        if now - getattr(self, "_last_clones", 0.0) > 3.0:
            clones = data.get("clones", [])
            self.populate_clones(clones)
            self._last_clones = now

        # presets table (throttled every ~5s)
        now = time.monotonic()
        if now - getattr(self, "_last_presets", 0.0) > 5.0:
            self.refresh_presets()
            self._last_presets = now


    # ---- Presets helpers/actions ----
    def _presets_selected_id(self):
        rows = self.presets_table.selectionModel().selectedRows()
        if not rows:
            return None, -1
        r = rows[0].row()
        item = self.presets_table.item(r, 1)
        pid = item.data(QtCore.Qt.UserRole) if item else None
        return pid, r

    def refresh_presets(self):
        data = api.get("/presets") or {}
        items = data.get("presets", []) if data.get("ok", False) else []
        self.presets_table.setRowCount(0)
        for p in items:
            r = self.presets_table.rowCount()
            self.presets_table.insertRow(r)
            star = "★" if p.get("starred") else "☆"
            it_star = QtWidgets.QTableWidgetItem(star)
            it_star.setTextAlignment(QtCore.Qt.AlignCenter)
            name = p.get("name") or "Preset"
            it_name = QtWidgets.QTableWidgetItem(name)
            it_name.setData(QtCore.Qt.UserRole, p.get("id"))
            self.presets_table.setItem(r, 0, it_star)
            self.presets_table.setItem(r, 1, it_name)

    def on_preset_new(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            name = time.strftime("Preset %Y-%m-%d %H:%M:%S")
        res = api.post("/presets", {"from_current": True, "name": name})
        if not res.get("ok", False):
            self.status.showMessage(f"Create preset error: {res.get('error','')}", 2000)
        self.refresh_presets()

    def on_preset_apply(self):
        pid, _ = self._presets_selected_id()
        if not pid:
            return
        res = api.post("/presets/apply", {"id": pid})
        ok = res.get("ok", False)
        self.status.showMessage("Preset applied" if ok else f"Apply error: {res.get('error','')}", 2000)
        # settings will refresh on next poll

    def on_preset_rename(self):
        pid, row = self._presets_selected_id()
        if not pid:
            return
        current = self.presets_table.item(row, 1).text()
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename Preset", "New name:", text=current)
        if not ok:
            return
        res = api.patch(f"/presets/{urllib.parse.quote(pid, safe='')}", {"name": name})
        ok = res.get("ok", False)
        self.status.showMessage("Renamed" if ok else f"Rename error: {res.get('error','')}", 2000)
        self.refresh_presets()

    def on_preset_delete(self):
        pid, _ = self._presets_selected_id()
        if not pid:
            return
        if QtWidgets.QMessageBox.question(self, "Delete Preset", "Delete selected preset?") != QtWidgets.QMessageBox.Yes:
            return
        res = api.delete(f"/presets/{urllib.parse.quote(pid, safe='')}")
        ok = res.get("ok", False)
        self.status.showMessage("Deleted" if ok else f"Delete error: {res.get('error','')}", 2000)
        self.refresh_presets()

    def on_preset_star(self):
        pid, _ = self._presets_selected_id()
        if not pid:
            return
        res = api.patch(f"/presets/{urllib.parse.quote(pid, safe='')}", {"starred": True})
        ok = res.get("ok", False)
        self.status.showMessage("Starred" if ok else f"Star error: {res.get('error','')}", 2000)
        self.refresh_presets()

    def on_preset_move(self, delta: int):
        data = api.get("/presets") or {}
        items = data.get("presets", []) if data.get("ok", False) else []
        if not items:
            return
        pid, row = self._presets_selected_id()
        if pid is None:
            return
        new_idx = max(0, min(len(items)-1, row + delta))
        if new_idx == row:
            return
        # Build new order array
        ids = [self.presets_table.item(r, 1).data(QtCore.Qt.UserRole) for r in range(self.presets_table.rowCount())]
        ids.insert(new_idx, ids.pop(row))
        res = api.post("/presets/reorder", {"order": ids})
        ok = res.get("ok", False)
        self.status.showMessage("Reordered" if ok else f"Reorder error: {res.get('error','')}", 2000)
        self.refresh_presets()
        # Reselect moved row
        self.presets_table.selectRow(new_idx)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()