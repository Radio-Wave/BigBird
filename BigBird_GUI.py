# BigBird_GUI_Qt.py
# PySide6 GUI for BigBird control server
# Requires: pip install PySide6 requests

from PySide6 import QtCore, QtGui, QtWidgets
import json
from pathlib import Path
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

    def patch(self, path, payload):
        try:
            r = requests.patch(self.base + path, json=payload, timeout=self.timeout)
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
        # (Test tone moved into Director Controls box below)

        # GUI overrides toggle — when off, backend ignores GUI/preset changes
        self.override_chk = QtWidgets.QCheckBox("Overwrite script settings")
        self.override_chk.setToolTip("When enabled, GUI and presets can change live TTS settings. When disabled, your script values stay authoritative.")
        self.override_chk.setChecked(False)  # default off: do not override on launch
        top.addWidget(self.override_chk)

        top.addStretch(1)

        # System info (cpu/mem/clones) + LED indicator
        self.sys_label = QtWidgets.QLabel("cpu: —   mem: —   clones: —")
        self.sys_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.sys_label)
        top.addSpacing(8)
        top.addWidget(QtWidgets.QLabel("LED:"))
        self.led_indicator = QtWidgets.QLabel()
        self.led_indicator.setFixedSize(14, 14)
        self.led_indicator.setToolTip("Current LED state")
        self._set_led_indicator_color("OFF")
        top.addWidget(self.led_indicator)

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
        self.tabs = QtWidgets.QTabWidget()
        mid.addWidget(self.tabs, 1)

        # --- Presets tab ---
        self.presets_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.presets_tab, "Presets")
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

        # --- PlayHT panel (will be embedded under Show Control) ---
        self.playht_tab = QtWidgets.QWidget()
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

        # --- ElevenLabs panel (will be embedded under Show Control) ---
        self.eleven_tab = QtWidgets.QWidget()
        eform = QtWidgets.QFormLayout(self.eleven_tab)

        self.e_boost = QtWidgets.QCheckBox("Enable speaker boost")
        eform.addRow(self.e_boost)
        self.e_stab  = add_row(eform, "Stability",        self._dbl(0.0, 1.5, 0.01, 0.5))
        self.e_sim   = add_row(eform, "Similarity boost", self._dbl(0.0, 1.5, 0.01, 0.8))
        self.e_style = add_row(eform, "Style",            self._dbl(0.0, 1.5, 0.01, 0.1))
        self.e_speed = add_row(eform, "Speed",            self._dbl(0.5, 2.0, 0.01, 1.0))

        self.apply_eleven = QtWidgets.QPushButton("Apply ElevenLabs")
        eform.addRow(self.apply_eleven)

        # --- VAD panel (will be embedded under Show Control) ---
        self.vad_tab = QtWidgets.QWidget()
        vform = QtWidgets.QFormLayout(self.vad_tab)

        self.v_mode = add_row(vform, "Aggressiveness (0–3)",
                              self._int(0, 3, 1, 2))
        self.v_init = add_row(vform, "Initial silence timeout (s)",
                              self._dbl(0.0, 60.0, 0.5, 15.0))
        self.v_sil  = add_row(vform, "Max trailing silence (s)",
                              self._dbl(0.0, 5.0, 0.1, 1.0))

        self.apply_vad = QtWidgets.QPushButton("Apply VAD")
        vform.addRow(self.apply_vad)

        # --- Arduino panel (will be embedded under Show Control) ---
        self.arduino_tab = QtWidgets.QWidget()
        aform = QtWidgets.QFormLayout(self.arduino_tab)
        self.ard_status = QtWidgets.QLabel("port: —   connected: —   override(hook/led): —/—")
        # Live hook state indicator for debugging
        self.hook_state_lbl = QtWidgets.QLabel("hook: —   raw: —")
        self.hook_state_lbl.setStyleSheet("color: #aaa;")
        aform.addRow(self.ard_status)
        aform.addRow(self.hook_state_lbl)

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

        # Ring controls (double buzz only)
        ring_row = QtWidgets.QHBoxLayout()
        ring_row.addWidget(QtWidgets.QLabel("Ring:"))
        self.ring_double_btn = QtWidgets.QPushButton("Double Buzz")
        self.ring_double_btn.clicked.connect(lambda: self.on_ring("double"))
        ring_row.addWidget(self.ring_double_btn)
        # Test tone button placed next to Double Buzz
        self.test_tone_btn = QtWidgets.QPushButton("Test Tone (1 kHz @ 44.1k)")
        self.test_tone_btn.setToolTip("Play a 1 kHz sine to verify audio routing")
        ring_row.addWidget(self.test_tone_btn)
        aform.addRow(ring_row)

        # --- Cloning panel (will be embedded under Show Control) ---
        self.cloning_tab = QtWidgets.QWidget()
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
        self.clone_total = QtWidgets.QLabel("total: 0.0s")
        info_row.addWidget(self.clone_total)
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

        # --- Show Control tab (consolidates PlayHT, ElevenLabs, VAD, Arduino, Cloning) ---
        self.show_tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self.show_tab)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        def make_box(title: str, content_widget: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            v.setContentsMargins(8, 6, 8, 8)
            v.addWidget(content_widget)
            return box

        box_playht = make_box("PlayHT", self.playht_tab)
        box_eleven = make_box("ElevenLabs", self.eleven_tab)
        box_vad    = make_box("VAD", self.vad_tab)

        # Director controls (admin override)
        self.director_panel = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(self.director_panel)
        self.btn_force_vad = QtWidgets.QPushButton("Trigger VAD End")
        self.btn_force_vad.setToolTip("Force end-of-speech now (cut recording)")
        drow.addWidget(self.btn_force_vad)
        self.btn_abort_tts = QtWidgets.QPushButton("Abort AI, Resume VAD")
        self.btn_abort_tts.setToolTip("Cut off current TTS playback and return to listening")
        drow.addWidget(self.btn_abort_tts)
        self.btn_end_conv = QtWidgets.QPushButton("End Conversation")
        self.btn_end_conv.setToolTip("Trigger the conversation finished flow")
        drow.addWidget(self.btn_end_conv)
        drow.addStretch(1)
        box_director = make_box("Director Controls", self.director_panel)

        box_ardu   = make_box("Arduino", self.arduino_tab)
        box_clone  = make_box("Cloning", self.cloning_tab)

        grid.addWidget(box_playht, 0, 0)
        grid.addWidget(box_eleven, 0, 1)
        grid.addWidget(box_vad, 0, 2)
        grid.addWidget(box_director, 1, 0, 1, 3)
        grid.addWidget(box_ardu, 2, 0, 1, 3)
        grid.addWidget(box_clone, 3, 0, 1, 3)

        self.tabs.addTab(self.show_tab, "Show Control")

        # --- Clones tab ---
        self.clones_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.clones_tab, "Clones")
        cbox = QtWidgets.QVBoxLayout(self.clones_tab)

        # Table
        self.clones_table = QtWidgets.QTableWidget(0, 6)
        self.clones_table.setHorizontalHeaderLabels(["Name", "ID", "Engine", "Alias", "Created", "Actions"])
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

        self.clones_sync_all_btn = QtWidgets.QPushButton("Sync All")
        self.clones_sync_all_btn.setToolTip("Fetch ElevenLabs voices for all aliases in keyring")
        self.clones_sync_all_btn.clicked.connect(self.on_clones_sync_all)
        cfooter.addWidget(self.clones_sync_all_btn)

        cfooter.addWidget(QtWidgets.QLabel("Evict oldest:"))
        self.clones_engine = QtWidgets.QComboBox()
        self.clones_engine.addItems(["All", "playht", "elevenlabs"])
        cfooter.addWidget(self.clones_engine)
        self.clones_evict_btn = QtWidgets.QPushButton("Evict")
        self.clones_evict_btn.clicked.connect(self.on_clones_evict)
        cfooter.addWidget(self.clones_evict_btn)

        # Alias filter
        cfooter.addSpacing(12)
        cfooter.addWidget(QtWidgets.QLabel("Alias:"))
        self.clones_alias_filter = QtWidgets.QComboBox()
        self.clones_alias_filter.addItems(["All", "Active"])  # dynamic aliases populated on render
        self.clones_alias_filter.currentIndexChanged.connect(self.on_clones_alias_filter)
        cfooter.addWidget(self.clones_alias_filter)

        cfooter.addStretch(1)
        cbox.addLayout(cfooter)

        # --- Logs tab ---
        self.logs_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.logs_tab, "Logs")
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
        self.test_tone_btn.clicked.connect(self.on_test_tone)
        self.apply_playht.clicked.connect(self.on_apply_playht)
        self.apply_eleven.clicked.connect(self.on_apply_eleven)
        self.apply_vad.clicked.connect(self.on_apply_vad)
        self.say_btn.clicked.connect(self.on_say)
        self.override_chk.stateChanged.connect(self.on_toggle_overrides)

        self.start_api_btn.clicked.connect(self.on_start_api)
        self.stop_api_btn.clicked.connect(self.on_stop_api)
        self.api_browse.clicked.connect(self.on_browse_api)
        self.clear_logs_btn.clicked.connect(self.logs.clear)
        self.clone_req_apply.clicked.connect(self.on_clone_threshold)
        self.clone_now_btn.clicked.connect(self.on_clone_now)
        
        # -------------------- Keys tab (ElevenLabs key manager) --------------------
        self.keys_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.keys_tab, "Keys")
        kroot = QtWidgets.QVBoxLayout(self.keys_tab)

        ktop = QtWidgets.QHBoxLayout()
        self.keys_filter = QtWidgets.QLineEdit()
        self.keys_filter.setPlaceholderText("Filter by alias…")
        self.keys_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.keys_add_btn = QtWidgets.QPushButton("Add Key")
        self.keys_edit_btn = QtWidgets.QPushButton("Edit Limit")
        self.keys_edit_used_btn = QtWidgets.QPushButton("Edit Used")
        self.keys_setactive_btn = QtWidgets.QPushButton("Set Active")
        self.keys_remove_btn = QtWidgets.QPushButton("Remove Selected")
        for w in (self.keys_filter, self.keys_refresh_btn, self.keys_add_btn, self.keys_edit_btn, self.keys_edit_used_btn, self.keys_setactive_btn, self.keys_remove_btn):
            ktop.addWidget(w)
        kroot.addLayout(ktop)

        self.keys_table = QtWidgets.QTableWidget(0, 8)
        self.keys_table.setHorizontalHeaderLabels([
            "Health", "Alias", "API Key", "IVC Limit", "Used", "Remaining", "Active", "Last Used",
        ])
        self.keys_table.horizontalHeader().setStretchLastSection(True)
        self.keys_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.keys_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        kroot.addWidget(self.keys_table, 1)

        self.keys_summary = QtWidgets.QLabel("")
        kroot.addWidget(self.keys_summary)

        # -------------------- Shortcuts tab --------------------
        self.shortcuts_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.shortcuts_tab, "Shortcuts")
        shot_root = QtWidgets.QVBoxLayout(self.shortcuts_tab)
        self.shortcuts_table = QtWidgets.QTableWidget(0, 2)
        self.shortcuts_table.setHorizontalHeaderLabels(["Action", "Keybinding"])
        self.shortcuts_table.horizontalHeader().setStretchLastSection(True)
        self.shortcuts_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.shortcuts_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        shot_root.addWidget(self.shortcuts_table, 1)
        shot_btns = QtWidgets.QHBoxLayout()
        self.shortcuts_apply_btn = QtWidgets.QPushButton("Apply")
        self.shortcuts_reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        shot_btns.addWidget(self.shortcuts_apply_btn)
        shot_btns.addWidget(self.shortcuts_reset_btn)
        shot_btns.addStretch(1)
        shot_root.addLayout(shot_btns)
        tip = QtWidgets.QLabel("Tip: On macOS, 'Meta' = Command. Shortcuts work when the app is focused.")
        tip.setStyleSheet("color:#777; font-size:11px")
        shot_root.addWidget(tip)

        # wire
        self.keys_refresh_btn.clicked.connect(self.keys_refresh)
        self.keys_filter.textChanged.connect(self.keys_refresh)
        self.keys_add_btn.clicked.connect(self.keys_add)
        self.keys_remove_btn.clicked.connect(self.keys_remove_selected)
        self.keys_setactive_btn.clicked.connect(self.keys_set_active)
        self.keys_edit_btn.clicked.connect(self.keys_edit_limit)
        self.keys_edit_used_btn.clicked.connect(self.keys_edit_used)
        # Director controls
        self.btn_force_vad.clicked.connect(self.on_director_force_vad)
        self.btn_abort_tts.clicked.connect(self.on_director_abort_tts)
        self.btn_end_conv.clicked.connect(self.on_director_end_conv)

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

        # Shortcuts tab handlers
        self.shortcuts_apply_btn.clicked.connect(self.on_shortcuts_apply)
        self.shortcuts_reset_btn.clicked.connect(self.on_shortcuts_reset)

        # Poll timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh_state)
        self.timer.start(800)  # ms

        # First refresh
        self.refresh_state()

        # Ensure buttons match the current checkbox until state poll confirms
        self._set_overrides_enabled(self.override_chk.isChecked())

        self._last_arduino = 0.0
        self._last_clones = 0.0
        self._last_presets = 0.0
        # Initialize keys view
        try:
            self.keys_refresh()
        except Exception:
            pass

        # Initialize keyboard shortcuts
        self._shortcuts = {}
        self._shortcuts_model = self._load_shortcuts()
        self._populate_shortcuts_table()
        self._bind_shortcuts()

    # ----------------- ElevenLabs keyring helpers -----------------
    @staticmethod
    def _keyring_path() -> Path:
        return Path.home() / ".bigbird" / "eleven_keyring.json"

    @staticmethod
    def _ensure_keyring():
        p = MainWindow._keyring_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            data = {"keys": [], "month_anchor": time.strftime("%Y-%m"), "last_active_alias": None}
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _load_keyring() -> dict:
        MainWindow._ensure_keyring()
        try:
            return json.loads(MainWindow._keyring_path().read_text(encoding="utf-8"))
        except Exception:
            return {"keys": [], "month_anchor": time.strftime("%Y-%m"), "last_active_alias": None}

    @staticmethod
    def _save_keyring(data: dict):
        MainWindow._ensure_keyring()
        MainWindow._keyring_path().write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _redact(key: str) -> str:
        if not key:
            return "—"
        if len(key) <= 10:
            return key[0:2] + "…" + key[-2:]
        return key[0:6] + "…" + key[-4:]

    @staticmethod
    def _rollover_month_if_needed(data: dict):
        cur = time.strftime("%Y-%m")
        if data.get("month_anchor") != cur:
            for k in data.get("keys", []):
                try:
                    k["ivc_used_this_month"] = 0
                except Exception:
                    pass
            data["month_anchor"] = cur

    # ----------------- Keys tab actions -----------------
    def keys_refresh(self):
        data = self._load_keyring()
        self._rollover_month_if_needed(data)
        keys = list(data.get("keys", []))
        flt = self.keys_filter.text().strip().lower()
        if flt:
            keys = [k for k in keys if flt in str(k.get("alias","")) .lower()]

        self.keys_table.setRowCount(0)
        for k in keys:
            alias = k.get("alias", "")
            api_key = k.get("api_key", "")
            limit = int(k.get("ivc_monthly_limit", 0) or 0)
            used = int(k.get("ivc_used_this_month", 0) or 0)
            remaining = max(0, limit - used)
            active = bool(k.get("active", False))
            last_used = float(k.get("last_used_ts", 0.0) or 0.0)
            last_str = time.strftime("%d/%m %H:%M", time.localtime(last_used)) if last_used > 0 else "—"

            row = self.keys_table.rowCount()
            self.keys_table.insertRow(row)
            # Health as colored circle
            color = "#c62828" if remaining <= 0 else ("#ef6c00" if (limit>0 and remaining/float(limit) < 0.2) else "#2e7d32")
            w = QtWidgets.QFrame()
            w.setFixedSize(14, 14)
            w.setStyleSheet(f"border-radius:7px; border:1px solid #555; background-color: {color};")
            self.keys_table.setCellWidget(row, 0, w)

            self.keys_table.setItem(row, 1, QtWidgets.QTableWidgetItem(alias))
            self.keys_table.setItem(row, 2, QtWidgets.QTableWidgetItem(self._redact(api_key)))
            self.keys_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(limit)))
            self.keys_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(used)))
            self.keys_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(remaining)))
            self.keys_table.setItem(row, 6, QtWidgets.QTableWidgetItem("Yes" if active else "No"))
            self.keys_table.setItem(row, 7, QtWidgets.QTableWidgetItem(last_str))

        # summary
        active_alias = next((k.get("alias") for k in data.get("keys", []) if k.get("active")), data.get("last_active_alias") or "—")
        active = next((k for k in data.get("keys", []) if k.get("active")), None)
        remaining = "—"
        if active:
            limit = int(active.get("ivc_monthly_limit", 0) or 0)
            used = int(active.get("ivc_used_this_month", 0) or 0)
            remaining = max(0, limit - used)
        self.keys_summary.setText(f"Active: {active_alias}  |  Remaining IVC: {remaining}  |  Keys: {len(data.get('keys', []))}")

    def _keys_selected_alias(self):
        sel = self.keys_table.selectionModel().selectedRows()
        if not sel:
            return None
        row = sel[0].row()
        it = self.keys_table.item(row, 1)
        return it.text() if it else None

    def keys_add(self):
        # Inline Add dialog
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Add ElevenLabs Key")
        alias = QtWidgets.QLineEdit()
        key = QtWidgets.QLineEdit()
        limit = QtWidgets.QSpinBox()
        limit.setRange(1, 100000)
        limit.setValue(95)
        active = QtWidgets.QCheckBox("Make active")
        form = QtWidgets.QFormLayout()
        form.addRow("Alias", alias)
        form.addRow("API Key", key)
        form.addRow("IVC Limit", limit)
        form.addRow("", active)
        btns = QtWidgets.QHBoxLayout()
        ok = QtWidgets.QPushButton("Add")
        cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        root = QtWidgets.QVBoxLayout(dlg)
        root.addLayout(form); root.addLayout(btns)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        a = alias.text().strip(); k = key.text().strip(); lim = int(limit.value()); make_active = bool(active.isChecked())
        if not a or not k:
            self.status.showMessage("Alias and API key required", 2000)
            return
        data = self._load_keyring()
        if any(r.get("alias") == a for r in data.get("keys", [])):
            self.status.showMessage("Alias already exists", 2000)
            return
        rec = {"alias": a, "api_key": k, "ivc_monthly_limit": lim, "ivc_used_this_month": 0, "active": False, "last_used_ts": 0.0, "voices_cache": {}}
        data.setdefault("keys", []).append(rec)
        if make_active or len(data["keys"]) == 1:
            for r in data["keys"]:
                r["active"] = (r.get("alias") == a)
                if r["active"]:
                    r["last_used_ts"] = time.time()
            data["last_active_alias"] = a
        self._rollover_month_if_needed(data)
        self._save_keyring(data)
        self.keys_refresh()

    def keys_remove_selected(self):
        alias = self._keys_selected_alias()
        if not alias:
            self.status.showMessage("Select a row to remove", 2000)
            return
        if QtWidgets.QMessageBox.question(self, "Confirm", f"Remove key '{alias}'?") != QtWidgets.QMessageBox.Yes:
            return
        data = self._load_keyring()
        keys = [k for k in data.get("keys", []) if k.get("alias") != alias]
        if data.get("last_active_alias") == alias:
            data["last_active_alias"] = keys[0]["alias"] if keys else None
        data["keys"] = keys
        self._save_keyring(data)
        self.keys_refresh()

    def keys_set_active(self):
        alias = self._keys_selected_alias()
        if not alias:
            self.status.showMessage("Select a row to set active", 2000)
            return
        data = self._load_keyring()
        found = False
        for k in data.get("keys", []):
            if k.get("alias") == alias:
                k["active"] = True
                k["last_used_ts"] = time.time()
                found = True
            else:
                k["active"] = False
        if not found:
            self.status.showMessage("Alias not found", 2000)
            return
        data["last_active_alias"] = alias
        self._save_keyring(data)
        self.keys_refresh()

    def keys_edit_limit(self):
        alias = self._keys_selected_alias()
        if not alias:
            self.status.showMessage("Select a row to edit", 2000)
            return
        data = self._load_keyring()
        rec = next((k for k in data.get("keys", []) if k.get("alias") == alias), None)
        if not rec:
            self.status.showMessage("Key not found", 2000)
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Edit IVC Limit — {alias}")
        spin = QtWidgets.QSpinBox(); spin.setRange(1, 100000); spin.setValue(int(rec.get("ivc_monthly_limit", 0) or 0))
        form = QtWidgets.QFormLayout(); form.addRow("IVC Limit", spin)
        btns = QtWidgets.QHBoxLayout(); ok = QtWidgets.QPushButton("Save"); cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept); cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        root = QtWidgets.QVBoxLayout(dlg); root.addLayout(form); root.addLayout(btns)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        rec["ivc_monthly_limit"] = int(spin.value())
        self._save_keyring(data)
        self.keys_refresh()

    def keys_edit_used(self):
        alias = self._keys_selected_alias()
        if not alias:
            self.status.showMessage("Select a row to edit used count", 2000)
            return
        data = self._load_keyring()
        rec = next((k for k in data.get("keys", []) if k.get("alias") == alias), None)
        if not rec:
            self.status.showMessage("Key not found", 2000)
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Edit Used This Month — {alias}")
        spin = QtWidgets.QSpinBox(); spin.setRange(0, 1000000); spin.setValue(int(rec.get("ivc_used_this_month", 0) or 0))
        form = QtWidgets.QFormLayout(); form.addRow("IVC Used", spin)
        btns = QtWidgets.QHBoxLayout(); ok = QtWidgets.QPushButton("Save"); cancel = QtWidgets.QPushButton("Cancel")
        ok.clicked.connect(dlg.accept); cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        root = QtWidgets.QVBoxLayout(dlg); root.addLayout(form); root.addLayout(btns)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        rec["ivc_used_this_month"] = int(spin.value())
        self._save_keyring(data)
        self.keys_refresh()
    def _set_led_indicator_color(self, state: str):
        colors = {
            "OFF": "#444",
            "RECORDING": "#d9534f",  # red
            "REPLYING": "#5bc0de",   # blue-ish
            "PROCESSING": "#f0ad4e", # amber
        }
        c = colors.get((state or "").upper(), "#444")
        self.led_indicator.setStyleSheet(
            f"border-radius: 7px; background-color: {c}; border: 1px solid #222;"
        )
    def on_director_force_vad(self):
        res = api.post("/control/vad/force-end", {}) or {}
        ok = res.get("ok", False)
        self.status.showMessage("VAD cut signaled" if ok else f"VAD cut error: {res.get('error','')}", 2000)

    def on_director_abort_tts(self):
        res = api.post("/control/abort-tts", {}) or {}
        ok = res.get("ok", False)
        self.status.showMessage("Playback abort signaled" if ok else f"Abort error: {res.get('error','')}", 2000)

    def on_director_end_conv(self):
        res = api.post("/control/conversation/end", {}) or {}
        ok = res.get("ok", False)
        self.status.showMessage("End conversation signaled" if ok else f"End error: {res.get('error','')}", 2000)
    def on_toggle_overrides(self, state):
        enabled = bool(state)
        res = api.post("/state/gui_overrides", {"enabled": enabled})
        ok = res.get("ok", False)
        self.status.showMessage(
            "GUI overrides " + ("enabled" if enabled else "disabled") if ok else f"Error: {res.get('error','')}",
            2000,
        )
        # Locally gray out Apply buttons if overrides are disabled
        self._set_overrides_enabled(enabled)

    def _set_overrides_enabled(self, enabled: bool):
        """Enable/disable controls that push settings to the backend."""
        for b in (
            self.apply_engine_btn,
            self.apply_audio_btn,
            self.apply_playht,
            self.apply_eleven,
            self.apply_vad,
            self.preset_apply_btn,
        ):
            b.setEnabled(bool(enabled))


    def on_hook_latch(self, which):
        # which: "auto" | "offhook" | "onhook"
        payload = {"state": None if which == "auto" else which}
        res = api.post("/arduino/hook", payload)
        ok = res.get("ok", False)
        self.status.showMessage("Hook set" if ok else f"Hook error: {res.get('error','')}", 2000)

    # ----------------- Shortcuts helpers -----------------
    @staticmethod
    def _shortcuts_path() -> Path:
        return Path.home() / ".bigbird" / "shortcuts.json"

    @staticmethod
    def _default_shortcuts() -> dict:
        # Use Qt portable sequences; users can customize in the Shortcuts tab
        import sys as _sys
        primary = "Meta" if _sys.platform == "darwin" else "Ctrl"
        return {
            "vad_force_end": f"{primary}+Shift+V",
            "abort_tts": f"{primary}+Shift+A",
            "end_conversation": f"{primary}+Shift+C",
            "hook_auto": f"{primary}+Alt+A",
            "hook_offhook": f"{primary}+Alt+O",
            "hook_onhook": f"{primary}+Alt+H",
            "ring_double_buzz": f"{primary}+Shift+R",
            "led_off": f"{primary}+Shift+0",
            "led_recording": f"{primary}+Shift+1",
            "led_replying": f"{primary}+Shift+2",
            "led_processing": f"{primary}+Shift+3",
        }

    @staticmethod
    def _ensure_shortcuts_file():
        p = MainWindow._shortcuts_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(json.dumps(MainWindow._default_shortcuts(), indent=2), encoding="utf-8")

    def _load_shortcuts(self) -> dict:
        try:
            self._ensure_shortcuts_file()
            data = json.loads(self._shortcuts_path().read_text(encoding="utf-8"))
            # merge with defaults in case new actions were added
            merged = dict(MainWindow._default_shortcuts())
            if isinstance(data, dict):
                merged.update({k: v for k, v in data.items() if isinstance(v, str)})
            return merged
        except Exception:
            return dict(MainWindow._default_shortcuts())

    def _save_shortcuts(self, data: dict):
        try:
            MainWindow._shortcuts_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _actions_catalog(self) -> list:
        # id, label, binder (callable to trigger)
        return [
            ("vad_force_end", "Trigger VAD End", self.on_director_force_vad),
            ("abort_tts", "Abort TTS", self.on_director_abort_tts),
            ("end_conversation", "End Conversation", self.on_director_end_conv),
            ("hook_auto", "Hook: Auto", lambda: (self.hook_auto_btn.setChecked(True), self.on_hook_latch("auto"))),
            ("hook_offhook", "Hook: Off-hook", lambda: (self.hook_off_btn.setChecked(True), self.on_hook_latch("offhook"))),
            ("hook_onhook", "Hook: On-hook", lambda: (self.hook_on_btn.setChecked(True), self.on_hook_latch("onhook"))),
            ("ring_double_buzz", "Ring: Double Buzz", lambda: self.on_ring("double")),
            ("led_off", "LED: OFF", lambda: self.on_led_set("OFF")),
            ("led_recording", "LED: RECORDING", lambda: self.on_led_set("RECORDING")),
            ("led_replying", "LED: REPLYING", lambda: self.on_led_set("REPLYING")),
            ("led_processing", "LED: PROCESSING", lambda: self.on_led_set("PROCESSING")),
        ]

    def _populate_shortcuts_table(self):
        self.shortcuts_table.setRowCount(0)
        data = self._shortcuts_model or {}
        for (aid, label, _handler) in self._actions_catalog():
            row = self.shortcuts_table.rowCount()
            self.shortcuts_table.insertRow(row)
            self.shortcuts_table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            # Key editor widget
            editor = QtWidgets.QKeySequenceEdit()
            seq_str = data.get(aid, "") or ""
            if seq_str:
                editor.setKeySequence(QtGui.QKeySequence(seq_str))
            editor.setProperty("action_id", aid)
            self.shortcuts_table.setCellWidget(row, 1, editor)

    def _bind_shortcuts(self):
        # Clear old
        for s in list(getattr(self, "_shortcuts", {}).values()):
            try:
                s.setParent(None)
            except Exception:
                pass
        self._shortcuts = {}

        for aid, _label, handler in self._actions_catalog():
            seq_str = (self._shortcuts_model or {}).get(aid, "") or ""
            if not seq_str:
                continue
            try:
                sc = QtWidgets.QShortcut(QtGui.QKeySequence(seq_str), self)
                sc.setContext(QtCore.Qt.ApplicationShortcut)
                sc.activated.connect(handler)
                self._shortcuts[aid] = sc
            except Exception:
                pass

    def on_shortcuts_apply(self):
        # Read current table values
        data = dict(self._shortcuts_model or {})
        rows = self.shortcuts_table.rowCount()
        for r in range(rows):
            w = self.shortcuts_table.cellWidget(r, 1)
            if isinstance(w, QtWidgets.QKeySequenceEdit):
                aid = w.property("action_id")
                seq = w.keySequence().toString(QtGui.QKeySequence.PortableText)
                data[str(aid)] = seq
        self._shortcuts_model = data
        self._save_shortcuts(data)
        self._bind_shortcuts()
        self.status.showMessage("Shortcuts applied", 2000)

    def on_shortcuts_reset(self):
        self._shortcuts_model = MainWindow._default_shortcuts()
        self._save_shortcuts(self._shortcuts_model)
        self._populate_shortcuts_table()
        self._bind_shortcuts()
        self.status.showMessage("Shortcuts reset to defaults", 2000)

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

    def on_test_tone(self):
        payload = {"seconds": 5, "freq": 1000.0, "rate": 44100, "device": "BlackHole"}
        res = api.post("/audio/test_tone", payload)
        if res.get("ok", False):
            dev = res.get("device", "?")
            rate = res.get("rate", "?")
            ch = res.get("channels", "?")
            self.status.showMessage(f"Tone sent to {dev} @ {rate} Hz ch {ch}", 2000)
        else:
            self.status.showMessage(f"Test tone error: {res.get('error','')}", 3000)

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
        # Only support double buzz from the GUI
        res = api.post("/arduino/ring", {"pattern": "double", "mode": "buzz"})
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
        led = ov.get("led", "OFF")
        self.ard_status.setText(f"port: {port}   connected: {conn}   override(hook/led): {hook}/{led}")
        # Live hook state/last raw for diagnosing
        hold = st.get("hold", None)
        hook_state = st.get("hook_state", None)
        raw = st.get("raw", None)
        disp = hook_state if hook_state else "—"
        raw_disp = raw if isinstance(raw, str) and raw.strip() else "—"
        self.hook_state_lbl.setText(f"hook: {disp}   raw: {raw_disp}")
        # Colorize: green for offhook, red for onhook, grey unknown
        if hook_state == "offhook":
            self.hook_state_lbl.setStyleSheet("color: #3c763d;")
        elif hook_state == "onhook":
            self.hook_state_lbl.setStyleSheet("color: #a94442;")
        else:
            self.hook_state_lbl.setStyleSheet("color: #aaa;")
        # update LED indicator
        self._set_led_indicator_color(led or "OFF")
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
        # Update alias filter options dynamically
        try:
            aliases = sorted({(c.get("account_alias") or "—") for c in clones if (c.get("engine") or "").lower() == "elevenlabs"})
        except Exception:
            aliases = []
        try:
            cur = self.clones_alias_filter.currentText() if hasattr(self, 'clones_alias_filter') else "All"
            items = ["All", "Active"] + [a for a in aliases if a not in ("All", "Active")]
            if hasattr(self, 'clones_alias_filter'):
                self.clones_alias_filter.blockSignals(True)
                self.clones_alias_filter.clear()
                self.clones_alias_filter.addItems(items)
                # restore selection if possible
                idx = self.clones_alias_filter.findText(cur)
                if idx >= 0:
                    self.clones_alias_filter.setCurrentIndex(idx)
                self.clones_alias_filter.blockSignals(False)
        except Exception:
            pass

        # Apply alias filter to ElevenLabs entries; PlayHT unaffected
        selected = None
        try:
            selected = self.clones_alias_filter.currentText()
        except Exception:
            selected = "All"
        active_alias = getattr(self, "_active_el_alias", "—")
        def _keep(c):
            eng = (c.get("engine") or "").lower()
            if eng != "elevenlabs":
                return True  # do not filter PlayHT
            if selected == "All":
                return True
            alias = c.get("account_alias") or "—"
            want = active_alias if selected == "Active" else selected
            return alias == want

        view = [c for c in clones if _keep(c)]

        self.clones_table.setRowCount(0)
        for c in view:
            row = self.clones_table.rowCount()
            self.clones_table.insertRow(row)
            name = c.get("name") or c.get("title") or "—"
            vid = c.get("id") or "—"
            eng = (c.get("engine") or c.get("provider") or "—").lower()
            alias = c.get("account_alias") or "—"
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
            it_alias = QtWidgets.QTableWidgetItem(str(alias))
            it_cr = QtWidgets.QTableWidgetItem(str(formatted_created))
            it_cr.setToolTip(str(raw_created))
            # stash id on the first item for convenience
            it_name.setData(QtCore.Qt.UserRole, vid)
            self.clones_table.setItem(row, 0, it_name)
            self.clones_table.setItem(row, 1, it_id)
            self.clones_table.setItem(row, 2, it_eng)
            self.clones_table.setItem(row, 3, it_alias)
            self.clones_table.setItem(row, 4, it_cr)

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
            self.clones_table.setCellWidget(row, 5, cell)

    def on_clones_sync_all(self):
        res = api.post("/clones/sync-all", {})
        ok = res.get("ok", False)
        self.status.showMessage("Clones synced (all aliases)" if ok else f"Sync-all error: {res.get('error','')}", 2000)
        self.refresh_clones_now()

    def on_clones_alias_filter(self):
        # Re-render table using cached items
        try:
            self.populate_clones(getattr(self, "_last_clones_items", []) or [])
        except Exception:
            pass

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
        # GUI overrides state (if backend exposes it)
        if isinstance(data.get("gui_overrides"), bool):
            ov = bool(data.get("gui_overrides"))
            self._set_if_idle(self.override_chk, self.override_chk.setChecked, ov)
            self._set_overrides_enabled(ov)

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
        ek = data.get("elevenlabs_keyring", {}) or {}
        ek_alias = ek.get("active_alias", "—")
        ek_rem = ek.get("remaining_ivc", "—")
        ek_str = f"   EL: {ek_alias} rem:{ek_rem}"
        self.sys_label.setText(f"cpu: {cpu}%   mem: {mem}   clones: {clones}{split_str}{ek_str}")
        # (Audio route debug removed)
        # remember active EL alias for filtering
        self._active_el_alias = ek.get("active_alias", "—") if isinstance(ek, dict) else "—"

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
        total_conv = float(cloning.get("conversation_seconds", 0.0) or 0.0)

        # Recording status: blend live elapsed into a greyed-out bar when active
        rec = data.get("recording", {}) or {}
        live_active = bool(rec.get("active", False))
        live_elapsed = float(rec.get("elapsed", 0.0) or 0.0)
        live_pct = pct
        if live_active and req > 0:
            live_pct = max(0.0, min(1.0, (col + live_elapsed) / req))

        if live_active:
            self.clone_bar.setStyleSheet("QProgressBar::chunk { background-color: #9aa0a6; }")
            self.clone_bar.setValue(int(max(0, min(100, round(live_pct * 100)))))
        else:
            self.clone_bar.setStyleSheet("")
            self.clone_bar.setValue(int(max(0, min(100, round(pct * 100)))))

        self.clone_info.setText(f"{col:.1f}/{req:.1f}s (rem {rem:.1f}s)")
        self.clone_counter.setText(f"clones this session: {sess}")
        self.clone_pending.setText(f"pending clips: {pend}")
        self.clone_total.setText(f"total: {total_conv:.1f}s")
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
            self._last_clones_items = list(clones)
            self.populate_clones(self._last_clones_items)
            self._last_clones = now

        # presets table (throttled every ~5s)
        now = time.monotonic()
        if now - getattr(self, "_last_presets", 0.0) > 5.0:
            self.refresh_presets()
            self._last_presets = now

        # keys tab auto-refresh if visible (every ~5s)
        try:
            if self.tabs.currentWidget() is self.keys_tab:
                if now - getattr(self, "_last_keys", 0.0) > 5.0:
                    self.keys_refresh()
                    self._last_keys = now
        except Exception:
            pass


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
