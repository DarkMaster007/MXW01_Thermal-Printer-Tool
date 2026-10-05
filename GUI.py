"""
MXW01 Thermal Printer GUI
-------------------------
Simple Tkinter front-end for MXW01printV3_clean.py.

What it supports:
- Print single image
- Print folder of images
- Print text
- Feed paper
- Debug-save prepared bitmaps instead of printing
- Select device by MAC or by BLE name
- Dither / threshold / overstrike / font / alignment / upside-down
- List fonts from the CLI

Run:
    python MXW01_GUI_clean.py
"""

import os
import queue
import subprocess
import sys
import threading
from tkinter import BOTH, END, E, N, NE, S, W, BooleanVar, IntVar, StringVar, Text, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


CLI_SCRIPT_NAME = "MXW01printV3.py"

# Try to get available fonts via matplotlib (used by the CLI)
_AVAILABLE_FONTS = []
try:
    from matplotlib import font_manager as _fm
    _AVAILABLE_FONTS = sorted(set(f.name for f in _fm.fontManager.ttflist))
except ImportError:
    pass
DITHER_CHOICES = [
    "none", "fs", "atkinson", "jarvis", "stucki", "burkes",
    "sierra", "sierra2", "sierra-lite", "bayer4", "bayer8", "dense"
]
ALIGN_CHOICES = ["left", "center", "right"]
MODE_CHOICES = [
    ("Image", "image"),
    ("Folder", "folder"),
    ("Text", "text"),
    ("Test print", "test"),
    ("Feed paper", "feed"),
]

TEST_IMAGE_FOLDER = "test_print_images"


class MXW01GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MXW01 Thermal Printer")
        self.proc = None
        self.output_q = queue.Queue()
        self.is_running = False
        self.after_id = None

        self.device_name = StringVar()
        self.device_addr = StringVar()
        self.mode = StringVar(value="image")
        self.image_path = StringVar()
        self.folder_path = StringVar()
        self.feed_lines = IntVar(value=40)
        self.font_name = StringVar(value=_AVAILABLE_FONTS[0] if _AVAILABLE_FONTS else "")
        self.font_size = IntVar(value=24)
        self.align = StringVar(value="left")
        self.dither = StringVar(value="fs")
        self.threshold = StringVar(value="auto")
        self.overstrike = IntVar(value=1)
        self.upside_down = BooleanVar(value=False)
        self.debug_save = BooleanVar(value=False)
        self.brightness = IntVar(value=128)
        self.intensity = IntVar(value=93)

        self._build_ui()
        self._on_mode_change()
        self.after_id = self.root.after(50, self._pump_output)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky=N + S + E + W)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        device_frame = ttk.LabelFrame(main, text="Device")
        device_frame.grid(row=0, column=0, columnspan=2, sticky=E + W, pady=(0, 8))
        device_frame.columnconfigure(1, weight=1)
        device_frame.columnconfigure(3, weight=1)

        ttk.Label(device_frame, text="Name:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        ttk.Entry(device_frame, textvariable=self.device_name).grid(row=0, column=1, sticky=E + W, padx=4, pady=4)
        ttk.Label(device_frame, text="MAC:").grid(row=0, column=2, sticky=E, padx=4, pady=4)
        ttk.Entry(device_frame, textvariable=self.device_addr).grid(row=0, column=3, sticky=E + W, padx=4, pady=4)

        mode_frame = ttk.LabelFrame(main, text="Mode")
        mode_frame.grid(row=1, column=0, sticky=N + S + E + W, pady=(0, 8))
        mode_frame.columnconfigure(0, weight=1)

        for row, (label, value) in enumerate(MODE_CHOICES):
            ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.mode, command=self._on_mode_change).grid(
                row=row, column=0, sticky=W, padx=4, pady=2
            )

        content_frame = ttk.LabelFrame(main, text="Content")
        content_frame.grid(row=1, column=1, sticky=N + S + E + W, pady=(0, 8))
        content_frame.columnconfigure(1, weight=1)

        ttk.Label(content_frame, text="Image:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        self.ent_img = ttk.Entry(content_frame, textvariable=self.image_path)
        self.ent_img.grid(row=0, column=1, sticky=E + W, padx=4, pady=4)
        self.btn_img = ttk.Button(content_frame, text="Browse...", command=self._pick_image)
        self.btn_img.grid(row=0, column=2, sticky=W, padx=4, pady=4)

        ttk.Label(content_frame, text="Folder:").grid(row=1, column=0, sticky=E, padx=4, pady=4)
        self.ent_dir = ttk.Entry(content_frame, textvariable=self.folder_path)
        self.ent_dir.grid(row=1, column=1, sticky=E + W, padx=4, pady=4)
        self.btn_dir = ttk.Button(content_frame, text="Browse...", command=self._pick_folder)
        self.btn_dir.grid(row=1, column=2, sticky=W, padx=4, pady=4)

        ttk.Label(content_frame, text="Text:").grid(row=2, column=0, sticky=NE, padx=4, pady=4)
        self.txt_text = Text(content_frame, height=5, wrap="word")
        self.txt_text.grid(row=2, column=1, columnspan=2, sticky=E + W, padx=4, pady=4)

        ttk.Label(content_frame, text="Feed lines:").grid(row=3, column=0, sticky=E, padx=4, pady=4)
        self.spn_feed = ttk.Spinbox(content_frame, from_=1, to=1000, increment=1, textvariable=self.feed_lines, width=8)
        self.spn_feed.grid(row=3, column=1, sticky=W, padx=4, pady=4)

        options_frame = ttk.LabelFrame(main, text="Options")
        options_frame.grid(row=2, column=0, columnspan=2, sticky=E + W, pady=(0, 8))
        for col in range(8):
            options_frame.columnconfigure(col, weight=1)

        ttk.Label(options_frame, text="Dither:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        ttk.Combobox(options_frame, textvariable=self.dither, values=DITHER_CHOICES, state="readonly").grid(
            row=0, column=1, sticky=E + W, padx=4, pady=4
        )

        ttk.Label(options_frame, text="Threshold:").grid(row=0, column=2, sticky=E, padx=4, pady=4)
        ttk.Entry(options_frame, textvariable=self.threshold, width=8).grid(row=0, column=3, sticky=W, padx=4, pady=4)

        ttk.Label(options_frame, text="Overstrike:").grid(row=0, column=4, sticky=E, padx=4, pady=4)
        ttk.Spinbox(options_frame, from_=1, to=3, increment=1, textvariable=self.overstrike, width=6).grid(
            row=0, column=5, sticky=W, padx=4, pady=4
        )

        self.chk_up = ttk.Checkbutton(options_frame, text="Upside down", variable=self.upside_down)
        self.chk_up.grid(row=0, column=6, sticky=W, padx=4, pady=4)
        self.chk_debug = ttk.Checkbutton(options_frame, text="Debug save", variable=self.debug_save)
        self.chk_debug.grid(row=0, column=7, sticky=W, padx=4, pady=4)

        ttk.Label(options_frame, text="Font:").grid(row=1, column=0, sticky=E, padx=4, pady=4)
        if _AVAILABLE_FONTS:
            ttk.Combobox(options_frame, textvariable=self.font_name, values=_AVAILABLE_FONTS, state="readonly").grid(
                row=1, column=1, sticky=E + W, padx=4, pady=4
            )
        else:
            ttk.Entry(options_frame, textvariable=self.font_name).grid(row=1, column=1, sticky=E + W, padx=4, pady=4)

        ttk.Label(options_frame, text="Size:").grid(row=1, column=2, sticky=E, padx=4, pady=4)
        ttk.Spinbox(options_frame, from_=6, to=96, increment=1, textvariable=self.font_size, width=6).grid(
            row=1, column=3, sticky=W, padx=4, pady=4
        )

        ttk.Label(options_frame, text="Align:").grid(row=1, column=4, sticky=E, padx=4, pady=4)
        ttk.Combobox(options_frame, textvariable=self.align, values=ALIGN_CHOICES, state="readonly").grid(
            row=1, column=5, sticky=E + W, padx=4, pady=4
        )

        ttk.Label(options_frame, text="Brightness:").grid(row=2, column=0, sticky=E, padx=4, pady=4)
        ttk.Spinbox(options_frame, from_=0, to=255, increment=1, textvariable=self.brightness, width=8).grid(
            row=2, column=1, sticky=W, padx=4, pady=4
        )

        ttk.Label(options_frame, text="Intensity:").grid(row=2, column=2, sticky=E, padx=4, pady=4)
        ttk.Spinbox(options_frame, from_=0, to=255, increment=1, textvariable=self.intensity, width=8).grid(
            row=2, column=3, sticky=W, padx=4, pady=4
        )

        action_frame = ttk.Frame(main)
        action_frame.grid(row=3, column=0, columnspan=2, sticky=E + W)
        action_frame.columnconfigure(0, weight=1)

        self.btn_run = ttk.Button(action_frame, text="Run", command=self._start_run)
        self.btn_run.grid(row=0, column=0, sticky=E, padx=4, pady=4)

        self.btn_fonts = ttk.Button(action_frame, text="List Fonts", command=self._list_fonts)
        self.btn_fonts.grid(row=0, column=1, sticky=E, padx=4, pady=4)

        self.btn_cancel = ttk.Button(action_frame, text="Cancel", command=self._cancel_run, state="disabled")
        self.btn_cancel.grid(row=0, column=2, sticky=E, padx=4, pady=4)

        self.log = ScrolledText(main, height=20, wrap="word")
        self.log.grid(row=4, column=0, columnspan=2, sticky=N + S + E + W, pady=(8, 0))
        main.rowconfigure(4, weight=1)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
        )
        if path:
            self.image_path.set(path)

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select folder")
        if path:
            self.folder_path.set(path)

    def _on_mode_change(self):
        mode = self.mode.get()

        image_enabled = mode == "image"
        folder_enabled = mode == "folder"
        text_enabled = mode == "text"
        test_enabled = mode == "test"
        feed_enabled = mode == "feed"

        self.ent_img.configure(state="normal" if image_enabled else "disabled")
        self.btn_img.configure(state="normal" if image_enabled else "disabled")
        self.ent_dir.configure(state="normal" if folder_enabled else "disabled")
        self.btn_dir.configure(state="normal" if folder_enabled else "disabled")
        self.spn_feed.configure(state="normal" if feed_enabled else "disabled")
        self.txt_text.configure(state="normal" if text_enabled else "disabled")

    def _append_log(self, text):
        self.log.insert(END, text)
        self.log.see(END)

    def _mask_cmd(self, cmd):
        return cmd

    def _require_cli(self):
        if not os.path.exists(CLI_SCRIPT_NAME):
            messagebox.showerror("Missing CLI", f"Cannot find '{CLI_SCRIPT_NAME}' in the current folder.")
            return False
        return True

    def _build_cmd(self, fonts_only=False):
        if not self._require_cli():
            return None

        cmd = [sys.executable, "-u", CLI_SCRIPT_NAME]

        if fonts_only:
            cmd.append("-l")
            return cmd

        device_name = self.device_name.get().strip()
        device_addr = self.device_addr.get().strip()
        if device_name:
            cmd += ["-N", device_name]
        elif device_addr:
            cmd += ["-d", device_addr]
        else:
            messagebox.showwarning("Device required", "Please enter a BLE device name or MAC address.")
            return None

        cmd += ["--dither", self.dither.get()]
        cmd += ["--overstrike", str(self.overstrike.get())]
        cmd += ["--threshold", self.threshold.get()]
        cmd += ["-n", self.font_name.get().strip() or "Arial"]
        cmd += ["-z", str(self.font_size.get())]
        cmd += ["-a", self.align.get()]
        cmd += ["--brightness", str(self.brightness.get())]
        cmd += ["--intensity", str(self.intensity.get())]

        if self.upside_down.get():
            cmd.append("-u")
        if self.debug_save.get():
            cmd.append("-s")

        mode = self.mode.get()
        if mode == "image":
            image = self.image_path.get().strip()
            if not image:
                messagebox.showwarning("Missing image", "Please select an image file.")
                return None
            cmd += ["-i", image]
        elif mode == "folder":
            folder = self.folder_path.get().strip()
            if not folder:
                messagebox.showwarning("Missing folder", "Please select a folder.")
                return None
            cmd += ["-f", folder]
        elif mode == "text":
            text_value = self.txt_text.get("1.0", "end-1c")
            if not text_value:
                messagebox.showwarning("Missing text", "Please enter text to print.")
                return None
            cmd += ["-t", text_value]
        elif mode == "test":
            if not os.path.isdir(TEST_IMAGE_FOLDER):
                messagebox.showwarning("Missing test folder", f"Could not find '{TEST_IMAGE_FOLDER}' next to the GUI.")
                return None
            cmd += ["-f", TEST_IMAGE_FOLDER]
        elif mode == "feed":
            cmd += ["-p", str(self.feed_lines.get())]
        else:
            messagebox.showerror("Invalid mode", f"Unknown mode: {mode}")
            return None

        return cmd

    def _run_proc(self, cmd):
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.output_q.put(line)

            returncode = self.proc.wait()
            self.output_q.put(f"\n[process exited with code {returncode}]\n")
        except Exception as exc:
            self.output_q.put(f"\n[failed to start process: {exc}]\n")
        finally:
            self.proc = None
            self.is_running = False
            self.output_q.put("__PROCESS_DONE__")

    def _start_run(self):
        if self.proc is not None:
            messagebox.showinfo("Busy", "A job is already running.")
            return

        cmd = self._build_cmd()
        if not cmd:
            return

        self._append_log("\n> " + " ".join(self._mask_cmd(cmd)) + "\n")
        self.is_running = True
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")

        thread = threading.Thread(target=self._run_proc, args=(cmd,), daemon=True)
        thread.start()

    def _list_fonts(self):
        if self.proc is not None:
            messagebox.showinfo("Busy", "A job is already running.")
            return

        cmd = self._build_cmd(fonts_only=True)
        if not cmd:
            return

        self._append_log("\n> " + " ".join(self._mask_cmd(cmd)) + "\n")
        self.is_running = True
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")

        thread = threading.Thread(target=self._run_proc, args=(cmd,), daemon=True)
        thread.start()

    def _cancel_run(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self._append_log("\n[termination requested]\n")
        except Exception as exc:
            self._append_log(f"\n[failed to terminate process: {exc}]\n")

    def _pump_output(self):
        try:
            while True:
                item = self.output_q.get_nowait()
                if item == "__PROCESS_DONE__":
                    self.btn_run.configure(state="normal")
                    self.btn_cancel.configure(state="disabled")
                    continue
                self._append_log(item)
        except queue.Empty:
            pass
        self.after_id = self.root.after(50, self._pump_output)

    def _on_close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        self.root.destroy()


def main():
    root = Tk()
    MXW01GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
