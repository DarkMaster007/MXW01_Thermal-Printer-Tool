"""
MX01W GUI Launcher (Tkinter)
----------------------------
A lightweight GUI that shells out to your existing CLI script (this keeps all your
Bleak/PIL logic untouched). No extra deps beyond Tkinter.

How it works:
- Builds the appropriate command line based on UI selections
- Runs your script in a background thread (subprocess) and streams logs to the UI
- Supports printing image/folder/text, test print, and paper feed
- Lets you pick dither mode, threshold, overstrike, font, alignment, upside-down, debug-save
- Specify device by name (preferred) or MAC address

Usage:
  1) Put this file next to your current CLI script, e.g. `catprint_cli.py`.
  2) Update CLI_SCRIPT_NAME below if your filename is different.
  3) Run:  python mx01w_gui.py

If you later refactor your CLI to expose a function instead of exiting with sys.exit,
we can switch from subprocess to direct function calls easily.
"""

import sys
import os
import threading
import subprocess
import queue
from tkinter import (
    Tk, StringVar, IntVar, BooleanVar, ttk, filedialog, messagebox, BOTH, END, N, S, E, W
)
from tkinter.scrolledtext import ScrolledText
import re

# === Configure the CLI script name here ===
CLI_SCRIPT_NAME = "MXW01printV2.py"  # change if your file is named differently

DITHER_CHOICES = [
    "none","fs","atkinson","jarvis","stucki","burkes",
    "sierra","sierra2","sierra-lite","bayer4","bayer8"
]
ALIGN_CHOICES = ["left", "center", "right"]

class MX01WGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MX01W Thermal Printer - GUI")
        self.proc = None
        self.output_q = queue.Queue()
        self.is_running = False

        # ====== State ======
        self.device_name = StringVar()
        self.device_addr = StringVar()
        self.mode = StringVar(value="image")  # image|folder|text|test|feed
        self.image_path = StringVar()
        self.folder_path = StringVar()
        self.text_to_print = StringVar()
        self.feed_lines = IntVar(value=40)
        self.font_name = StringVar(value="Arial")
        self.font_size = IntVar(value=24)
        self.align = StringVar(value="left")
        self.dither = StringVar(value="fs")
        self.threshold = StringVar(value="auto")
        self.overstrike = IntVar(value=1)
        self.upside_down = BooleanVar(value=False)
        self.debug_save = BooleanVar(value=False)

        # ====== Layout ======
        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky=N+S+E+W)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Device frame
        devf = ttk.LabelFrame(main, text="Device")
        devf.grid(row=0, column=0, columnspan=2, sticky=E+W, pady=(0,8))
        devf.columnconfigure(1, weight=1)
        devf.columnconfigure(3, weight=1)
        ttk.Label(devf, text="Name (preferred):").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        ttk.Entry(devf, textvariable=self.device_name).grid(row=0, column=1, sticky=E+W, padx=4, pady=4)
        ttk.Label(devf, text="MAC addr:").grid(row=0, column=2, sticky=E, padx=4, pady=4)
        ttk.Entry(devf, textvariable=self.device_addr).grid(row=0, column=3, sticky=E+W, padx=4, pady=4)

        # Mode frame
        modef = ttk.LabelFrame(main, text="Mode")
        modef.grid(row=1, column=0, sticky=N+S+E+W, pady=(0,8))
        for i in range(2):
            modef.columnconfigure(i, weight=1)
        # radio buttons
        modes = [
            ("Image", "image"),
            ("Folder", "folder"),
            ("Text", "text"),
            ("Test print", "test"),
            ("Feed paper", "feed"),
        ]
        for r, (label, val) in enumerate(modes):
            ttk.Radiobutton(modef, text=label, value=val, variable=self.mode, command=self._on_mode_change).grid(row=r, column=0, sticky=W, padx=4, pady=2)

        # Content frame
        contf = ttk.LabelFrame(main, text="Content")
        contf.grid(row=1, column=1, sticky=N+S+E+W, pady=(0,8))
        contf.columnconfigure(1, weight=1)

        # image / folder selectors
        self.btn_img = ttk.Button(contf, text="Select Image…", command=self._pick_image)
        self.ent_img = ttk.Entry(contf, textvariable=self.image_path)
        self.btn_dir = ttk.Button(contf, text="Select Folder…", command=self._pick_folder)
        self.ent_dir = ttk.Entry(contf, textvariable=self.folder_path)

        # text entry
        ttk.Label(contf, text="Text:").grid(row=2, column=0, sticky=E, padx=4, pady=4)
        self.ent_text = ttk.Entry(contf, textvariable=self.text_to_print)
        self.ent_text.grid(row=2, column=1, sticky=E+W, padx=4, pady=4)

        # feed
        ttk.Label(contf, text="Feed lines:").grid(row=3, column=0, sticky=E, padx=4, pady=4)
        self.spn_feed = ttk.Spinbox(contf, from_=1, to=500, increment=1, textvariable=self.feed_lines, width=8)
        self.spn_feed.grid(row=3, column=1, sticky=W, padx=4, pady=4)

        # place img/folder rows
        ttk.Label(contf, text="Image:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        self.ent_img.grid(row=0, column=1, sticky=E+W, padx=4, pady=4)
        self.btn_img.grid(row=0, column=2, sticky=W, padx=4, pady=4)
        ttk.Label(contf, text="Folder:").grid(row=1, column=0, sticky=E, padx=4, pady=4)
        self.ent_dir.grid(row=1, column=1, sticky=E+W, padx=4, pady=4)
        self.btn_dir.grid(row=1, column=2, sticky=W, padx=4, pady=4)

        # Options frame
        optf = ttk.LabelFrame(main, text="Options")
        optf.grid(row=2, column=0, columnspan=2, sticky=E+W, pady=(0,8))
        for c in range(8):
            optf.columnconfigure(c, weight=1)

        ttk.Label(optf, text="Dither:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        ttk.Combobox(optf, textvariable=self.dither, values=DITHER_CHOICES, state="readonly").grid(row=0, column=1, sticky=E+W, padx=4, pady=4)

        ttk.Label(optf, text="Threshold:").grid(row=0, column=2, sticky=E, padx=4, pady=4)
        ttk.Entry(optf, textvariable=self.threshold, width=8).grid(row=0, column=3, sticky=W, padx=4, pady=4)

        ttk.Label(optf, text="Overstrike:").grid(row=0, column=4, sticky=E, padx=4, pady=4)
        ttk.Spinbox(optf, from_=1, to=3, increment=1, textvariable=self.overstrike, width=6).grid(row=0, column=5, sticky=W, padx=4, pady=4)

        ttk.Label(optf, text="Font:").grid(row=1, column=0, sticky=E, padx=4, pady=4)
        ttk.Entry(optf, textvariable=self.font_name).grid(row=1, column=1, sticky=E+W, padx=4, pady=4)

        ttk.Label(optf, text="Size:").grid(row=1, column=2, sticky=E, padx=4, pady=4)
        ttk.Spinbox(optf, from_=6, to=96, increment=1, textvariable=self.font_size, width=6).grid(row=1, column=3, sticky=W, padx=4, pady=4)

        ttk.Label(optf, text="Align:").grid(row=1, column=4, sticky=E, padx=4, pady=4)
        ttk.Combobox(optf, textvariable=self.align, values=ALIGN_CHOICES, state="readonly").grid(row=1, column=5, sticky=E+W, padx=4, pady=4)

        self.chk_up = ttk.Checkbutton(optf, text="Upside down", variable=self.upside_down)
        self.chk_up.grid(row=0, column=6, sticky=W, padx=4, pady=4)
        self.chk_dbg = ttk.Checkbutton(optf, text="Debug save", variable=self.debug_save)
        self.chk_dbg.grid(row=0, column=7, sticky=W, padx=4, pady=4)

        # Actions frame
        actf = ttk.Frame(main)
        actf.grid(row=3, column=0, columnspan=2, sticky=E+W)
        actf.columnconfigure(0, weight=1)
        actf.columnconfigure(1, weight=0)
        actf.columnconfigure(2, weight=0)
        self.btn_run = ttk.Button(actf, text="Print / Run", command=self._start_run)
        self.btn_run.grid(row=0, column=0, sticky=E, padx=4, pady=4)
        self.btn_list_fonts = ttk.Button(actf, text="List Fonts", command=self._list_fonts)
        self.btn_list_fonts.grid(row=0, column=1, sticky=E, padx=4, pady=4)
        self.btn_cancel = ttk.Button(actf, text="Cancel", command=self._cancel_run, state="disabled")
        self.btn_cancel.grid(row=0, column=2, sticky=E, padx=4, pady=4)

        # Log output
        self.log = ScrolledText(main, height=20, wrap="word")
        self.log.grid(row=4, column=0, columnspan=2, sticky=N+S+E+W, pady=(8,0))
        main.rowconfigure(4, weight=1)

        self._on_mode_change()
        self.root.after(50, self._pump_output)

    # ====== UI handlers ======
    def _on_mode_change(self):
        mode = self.mode.get()
        # Enable/disable content controls based on mode
        image_enabled = (mode == "image")
        folder_enabled = (mode == "folder")
        text_enabled = (mode == "text")
        feed_enabled = (mode == "feed")

        # Image
        state = "normal" if image_enabled else "disabled"
        self.ent_img.configure(state=state)
        self.btn_img.configure(state=state)
        # Folder
        state = "normal" if folder_enabled else "disabled"
        self.ent_dir.configure(state=state)
        self.btn_dir.configure(state=state)
        # Text
        self.ent_text.configure(state=("normal" if text_enabled else "disabled"))
        # Feed
        self.spn_feed.configure(state=("normal" if feed_enabled else "disabled"))

    def _pick_image(self):
        path = filedialog.askopenfilename(title="Select image",
                                          filetypes=[("Images", ".png .jpg .jpeg .bmp .gif"), ("All", "*.*")])
        if path:
            self.image_path.set(path)

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select folder with images")
        if path:
            self.folder_path.set(path)

    # ====== Subprocess orchestration ======
    def _build_cmd(self, mode_override=None):
        """Build the CLI command based on UI state."""
        if not os.path.exists(CLI_SCRIPT_NAME):
            messagebox.showerror("Missing script", f"Cannot find '{CLI_SCRIPT_NAME}' next to this file.")
            return None

        cmd = [sys.executable, "-u", CLI_SCRIPT_NAME]
        # device
        if self.device_name.get().strip():
            cmd += ["-N", self.device_name.get().strip()]
        elif self.device_addr.get().strip():
            cmd += ["-d", self.device_addr.get().strip()]
        else:
            # allow --list-fonts without device, but otherwise require
            if (mode_override or self.mode.get()) != "fonts":
                messagebox.showwarning("Device required", "Please enter a Device Name or MAC address.")
                return None

        # options
        if self.debug_save.get():
            cmd += ["-s"]
        if self.upside_down.get():
            cmd += ["-u"]
        cmd += ["--dither", self.dither.get()]
        cmd += ["--overstrike", str(self.overstrike.get())]
        cmd += ["--threshold", self.threshold.get()]

        # font options (used only when -t)
        cmd += ["-n", self.font_name.get().strip() or "Arial"]
        cmd += ["-z", str(self.font_size.get())]
        cmd += ["-a", self.align.get()]

        mode = mode_override or self.mode.get()
        if mode == "image":
            if not self.image_path.get().strip():
                messagebox.showwarning("Select image", "Please choose an image file.")
                return None
            cmd += ["-i", self.image_path.get().strip()]
        elif mode == "folder":
            if not self.folder_path.get().strip():
                messagebox.showwarning("Select folder", "Please choose a folder.")
                return None
            cmd += ["-f", self.folder_path.get().strip()]
        elif mode == "text":
            txt = self.text_to_print.get()
            if not txt:
                messagebox.showwarning("Enter text", "Please enter the text to print.")
                return None
            cmd += ["-t", txt]
        elif mode == "test":
            cmd += ["-x"]
        elif mode == "feed":
            cmd += ["-p", str(self.feed_lines.get())]
        elif mode == "fonts":
            cmd = [sys.executable, CLI_SCRIPT_NAME, "-l"]
        else:
            messagebox.showerror("Invalid mode", f"Unknown mode: {mode}")
            return None
        return cmd

    def _start_run(self):
        cmd = self._build_cmd()
        if not cmd:
            return
        if self.proc is not None:
            messagebox.showinfo("Busy", "A job is already running. Please cancel or wait.")
            return
        self._append_log(f"\n> {' '.join(self._mask_cmd(cmd))}\n")
        self.is_running = True
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        t = threading.Thread(target=self._run_proc, args=(cmd,), daemon=True)
        t.start()

    def _ask_preview_choice_blocking(self, desc: str) -> str:
        done = threading.Event()
        result = {"ans": "y"}

        def show_dialog():
            win = ttk.Toplevel(self.root)
            win.title("Confirm print"); win.transient(self.root); win.grab_set()
            ttk.Label(win, text=f"Print this job?\n{desc}").grid(row=0, column=0, columnspan=4, padx=12, pady=(12,8))
            def choose(ch):
                result["ans"] = ch
                try: win.grab_release()
                except Exception: pass
                win.destroy(); done.set()
            ttk.Button(win, text="Yes", command=lambda: choose('y')).grid(row=1, column=0, padx=6, pady=10)
            ttk.Button(win, text="Skip", command=lambda: choose('s')).grid(row=1, column=1, padx=6, pady=10)
            ttk.Button(win, text="All remaining", command=lambda: choose('a')).grid(row=1, column=2, padx=6, pady=10)
            ttk.Button(win, text="Quit", command=lambda: choose('q')).grid(row=1, column=3, padx=6, pady=10)
            for c in range(4): win.columnconfigure(c, weight=1)
            win.protocol("WM_DELETE_WINDOW", lambda: choose('s'))
            win.geometry("420x130")
        self.root.after(0, show_dialog)
        done.wait()
        return result["ans"]

    def _run_proc(self, cmd):
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,   # << allow replying to the CLI
                text=True,
                bufsize=1
            )

            # detect the CLI interactive preview prompt
            prompt_re = re.compile(
                r"^\[(\d+)/(\d+)\] Print '(.+?)'\? \[Y\]es / \[s\]kip / \[a\]ll remaining / \[q\]uit:"
            )

            for raw in self.proc.stdout:
                line = raw.rstrip("\n")

                # If the CLI prints its question without a trailing newline, we still
                # want to catch it. The -u flag above helps ensure we get it promptly.
                m = prompt_re.search(line)
                if m and self.proc and self.proc.stdin:
                    desc = m.group(3)
                    ans = self._ask_preview_choice_blocking(desc)   # 'y'/'s'/'a'/'q'
                    try:
                        self.proc.stdin.write(ans + "\n")
                        self.proc.stdin.flush()
                    except Exception as e:
                        self.output_q.put(f"\n[Error sending response to CLI] {e}\n")

                # forward output to the log
                self.output_q.put(raw)

            self.proc.wait()
            rc = self.proc.returncode
            self.output_q.put(f"\n[Process exited with code {rc}]\n")

        except Exception as e:
            self.output_q.put(f"\n[Error] {e}\n")
        finally:
            self.proc = None
            self.is_running = False

    def _cancel_run(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.btn_cancel.configure(state="disabled")
        self.btn_run.configure(state="normal")

    def _list_fonts(self):
        cmd = self._build_cmd(mode_override="fonts")
        if not cmd:
            return
        self._append_log(f"\n> {' '.join(cmd)}\n")
        threading.Thread(target=self._run_proc, args=(cmd,), daemon=True).start()

    def _pump_output(self):
        while True:
            try:
                line = self.output_q.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        # buttons state if proc ended
        if not self.is_running:
            self.btn_cancel.configure(state="disabled")
            self.btn_run.configure(state="normal")
        self.root.after(50, self._pump_output)

    def _append_log(self, text):
        self.log.insert(END, text)
        self.log.see(END)

    def _mask_cmd(self, cmd_list):
        """Mask potentially long text args for display (e.g., -t).
        Keep it readable in the log."""
        masked = []
        it = iter(enumerate(cmd_list))
        for i, tok in it:
            if tok in ("-t", "--text"):
                masked.append(tok)
                try:
                    _i, val = next(it)
                    if len(val) > 60:
                        val = val[:57] + "…"
                    masked.append(val)
                except StopIteration:
                    pass
            else:
                masked.append(tok)
        return masked

if __name__ == "__main__":
    root = Tk()
    # a bit of modern-ish styling
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # crisp on Windows, ignore elsewhere
    except Exception:
        pass
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = MX01WGUI(root)
    root.geometry("980x640")
    root.mainloop()
