"""
MXW01 Thermal Printer GUI
-------------------------
Simple Tkinter front-end using persistent session mode with MXW01printV3.py.

What it supports:
- Print single image
- Print folder of images
- Print text
- Feed paper
- Debug-save prepared bitmaps instead of printing
- Select device by MAC or by BLE name
- Dither / threshold / overstrike / font / alignment / upside-down
- List fonts from the CLI
- Persistent connection — no re-connecting between jobs

Run:
    python GUI.py
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
        
        # Session management
        self.proc = None  # Subprocess handle for the CLI session
        self.output_q = queue.Queue()
        self.is_connected = False  # Whether CLI is connected and ready
        self.read_thread = None
        
        # Device settings
        self.device_name = StringVar()
        self.device_addr = StringVar()
        
        # Mode/content settings
        self.mode = StringVar(value="image")
        self.image_path = StringVar()
        self.folder_path = StringVar()
        self.feed_lines = IntVar(value=40)
        
        # Format options
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
        
        # Device frame
        device_frame = ttk.LabelFrame(main, text="Device")
        device_frame.grid(row=0, column=0, columnspan=2, sticky=E + W, pady=(0, 8))
        device_frame.columnconfigure(1, weight=1)
        device_frame.columnconfigure(3, weight=1)
        
        ttk.Label(device_frame, text="Name:").grid(row=0, column=0, sticky=E, padx=4, pady=4)
        ttk.Entry(device_frame, textvariable=self.device_name).grid(row=0, column=1, sticky=E + W, padx=4, pady=4)
        ttk.Label(device_frame, text="MAC:").grid(row=0, column=2, sticky=E, padx=4, pady=4)
        ttk.Entry(device_frame, textvariable=self.device_addr).grid(row=0, column=3, sticky=E + W, padx=4, pady=4)
        
        # Mode frame
        mode_frame = ttk.LabelFrame(main, text="Mode")
        mode_frame.grid(row=1, column=0, sticky=N + S + E + W, pady=(0, 8))
        mode_frame.columnconfigure(0, weight=1)
        
        for row, (label, value) in enumerate(MODE_CHOICES):
            ttk.Radiobutton(mode_frame, text=label, value=value, variable=self.mode, command=self._on_mode_change).grid(
                row=row, column=0, sticky=W, padx=4, pady=2
            )
        
        # Content frame
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
        
        # Options frame
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
        
        # Action buttons
        action_frame = ttk.Frame(main)
        action_frame.grid(row=3, column=0, columnspan=2, sticky=E + W)
        action_frame.columnconfigure(0, weight=1)
        
        self.btn_run = ttk.Button(action_frame, text="Start Session", command=self._start_session)
        self.btn_run.grid(row=0, column=0, sticky=E, padx=4, pady=4)
        
        self.btn_fonts = ttk.Button(action_frame, text="List Fonts", command=self._list_fonts)
        self.btn_fonts.grid(row=0, column=1, sticky=E, padx=4, pady=4)
        
        self.btn_cancel = ttk.Button(action_frame, text="Disconnect", command=self._stop_session, state="disabled")
        self.btn_cancel.grid(row=0, column=2, sticky=E, padx=4, pady=4)
        
        # Log area & status bar
        self.log = ScrolledText(main, height=20, wrap="word")
        self.log.grid(row=4, column=0, columnspan=2, sticky=N + S + E + W, pady=(8, 0))
        main.rowconfigure(4, weight=1)
        
        self.status_label = ttk.Label(main, text="Status: Not connected", foreground="red")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=(4, 0), sticky=W)
    
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
    
    def _update_status(self, connected=True, message=None):
        """Update status label and button states."""
        if not hasattr(self, 'status_label'):
            return
        
        if connected:
            self.status_label.config(text=message or "Status: Connected & Ready", foreground="green")
            # Toggle btn_run to "Run" for printing jobs
            self.btn_run.config(text="Run", command=self._start_run)
            self.btn_run.configure(state="normal")
            self.btn_cancel.config(state="normal")
            self.btn_fonts.config(state="normal")
            self.is_connected = True
        else:
            self.status_label.config(text=message or "Status: Disconnected", foreground="red")
            # Reset btn_run to "Start Session"
            self.btn_run.config(text="Start Session", command=self._start_session)
            self.btn_run.configure(state="normal")
            self.btn_cancel.config(state="disabled")
            self.btn_fonts.config(state="normal")
            self.is_connected = False
    
    def _require_cli(self):
        if not os.path.exists(CLI_SCRIPT_NAME):
            messagebox.showerror("Missing CLI", f"Cannot find '{CLI_SCRIPT_NAME}' in the current folder.")
            return False
        return True
    
    def _send_command(self, cmd_line):
        """Send a command line to the running session process via stdin."""
        if self.proc is None or self.proc.poll() is not None:
            self._append_log("\n[ERROR: No active session]\n")
            return False
        
        try:
            self.proc.stdin.write(cmd_line + "\n")
            self.proc.stdin.flush()
            return True
        except Exception as exc:
            self._append_log(f"\n[Failed to send command: {exc}]\n")
            return False
    
    def _start_session(self):
        """Launch CLI in session mode and wait for connection confirmation."""
        if not self._require_cli():
            return
        
        # Check device info
        device_name = self.device_name.get().strip()
        device_addr = self.device_addr.get().strip()
        
        if not device_name and not device_addr:
            messagebox.showwarning("Device required", "Please enter a BLE device name or MAC address before starting session.")
            return
        
        # Build session launch command
        cmd = [sys.executable, "-u", CLI_SCRIPT_NAME, "--session"]
        if device_name:
            cmd.extend(["-N", device_name])
        elif device_addr:
            cmd.extend(["-d", device_addr])
        
        self._append_log(f"> Starting session with: {' '.join(cmd)}\n")
        
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
                text=True,
            )
            
            # Start reader thread
            self.read_thread = threading.Thread(target=self._read_stdin, daemon=True)
            self.read_thread.start()
            
            # Wait for definitive status without arbitrary short timeouts.
            # BLE scanning can legitimately take 10–30s depending on radio state.
            import time
            start_time = time.time()
            max_wait = 120  # Safety ceiling so we never hang forever
            
            while True:
                elapsed = time.time() - start_time
                
                # Safety ceiling — force kill if nothing happened for too long
                if elapsed > max_wait:
                    self._append_log(f"\n[Safety timeout after {int(elapsed)}s — forcing disconnect]\n")
                    messagebox.showwarning("Timeout", "Printer did not respond within 2 minutes. Process will be terminated.")
                    self._stop_session()
                    return
                
                # Check process exit
                if self.proc.poll() is not None:
                    break
                
                # Drain output queue and look for definitive signals
                temp_lines = []
                while True:
                    try:
                        item = self.output_q.get_nowait()
                        temp_lines.append(item)
                    except queue.Empty:
                        break
                
                # Success — immediately transition UI
                if any("Connected to" in l for l in temp_lines):
                    for line in temp_lines:
                        self._append_log(line)
                    self._update_status(connected=True, message="Status: Connected")
                    self._append_log("\n--- Session active. Select mode and press Run to print ---\n")
                    return
                
                # Clear failure — show error and cleanup
                if any("Failed to connect" in l.lower() for l in temp_lines):
                    for line in temp_lines:
                        self._append_log(line)
                    messagebox.showerror("Connection Failed", "Could not connect to printer. Is it powered on?")
                    self._stop_session()
                    return
                
                if any("Error resolving device" in l for l in temp_lines):
                    for line in temp_lines:
                        self._append_log(line)
                    messagebox.showerror("Discovery Error", "BLE scanner could not find the device.\nMake sure it is powered on and nearby.")
                    self._stop_session()
                    return
                
                # Brief sleep, keep polling
                time.sleep(0.25)
            
            # Process exited — drain ALL remaining buffered output
            extra_output = []
            try:
                while True:
                    extra_output.append(self.output_q.get_nowait())
            except queue.Empty:
                pass
            
            if extra_output:
                # Log everything first so user sees what happened
                for line in extra_output:
                    self._append_log(line)
                
                # Then check final result
                if any("Connected to" in l for l in extra_output):
                    self._update_status(connected=True, message="Status: Connected")
                    self._append_log("\n--- Session active. Select mode and press Run to print ---\n")
                    return
                
            # Process exited without showing success
            retcode = self.proc.returncode
            if retcode != 0:
                self._append_log(f"\n[Session exited with error code {retcode}]\n")
                messagebox.showerror("Session Error", f"CLI exited unexpectedly (code {retcode}).")
            else:
                self._append_log(f"\n[Session ended — process exited cleanly (code 0)]\n")
            
            self._stop_session()
            
        except Exception as exc:
            self._append_log(f"\n[Failed to start session: {exc}]\n")
            messagebox.showerror("Error", str(exc))
    
    def _read_stdin(self):
        """Read output from CLI session and queue it for GUI display."""
        assert self.proc is not None and self.proc.stdout is not None
        
        while True:
            try:
                line = self.proc.stdout.readline()
                if not line:  # EOF
                    break
                self.output_q.put(line)
            except Exception:
                break
        
        # Process ended
        returncode = self.proc.wait() if self.proc.poll() is None else self.proc.returncode
        self.output_q.put(f"\n[Session exited with code {returncode}]\n")
        
        # Signal done
        self.output_q.put("__PROCESS_DONE__")
        
        # Cleanup
        self.proc = None
        self._update_status(connected=False, message="Status: Disconnected")
    
    def _stop_session(self):
        """Send quit command and disconnect gracefully."""
        if self.proc is None:
            self._update_status(connected=False)
            return
        
        self._append_log("\n[Stopping session...]")
        self._send_command("quit")
        
        # Give it a moment to shut down cleanly via quit command
        import time
        deadline = time.time() + 3
        while time.time() < deadline and self.proc.poll() is None:
            time.sleep(0.2)
        
        if self.proc.poll() is None:
            # SIGTERM doesn't trigger Python KeyboardInterrupt → BLE disconnect won't run.
            # Use SIGINT (Ctrl+C) so the session's except KeyboardInterrupt: block fires,
            # which contains the clean BLE stop_notify/disconnect cleanup.
            self._append_log("[Force interrupting with SIGINT...]")
            try:
                import signal
                self.proc.send_signal(signal.SIGINT)
            except Exception as e:
                self._append_log(f"[SIGINT failed: {e}], falling back to terminate...")
                self.proc.terminate()
            
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self._append_log("[Still alive — killing.]")
                self.proc.kill()
        
        self._cleanup()
    
    def _cleanup(self):
        """Clean up subprocess resources."""
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1)
            except Exception:
                pass
        self.proc = None
        self._update_status(connected=False)
    
    def _list_fonts(self):
        """List available fonts via session."""
        if not self.is_connected:
            messagebox.showinfo("Info", "Start a session first before listing fonts.")
            return
        
        self._append_log("\n> Listing fonts...")
        self._send_command("fonts")
        self._append_log("(See output below)\n")
    
    def _start_run(self):
        """Execute selected action via the running session."""
        if not self.is_connected:
            messagebox.showinfo("Info", "No active session. Click 'Start Session' first.")
            return
        
        device_name = self.device_name.get().strip()
        device_addr = self.device_addr.get().strip()
        
        if not device_name and not device_addr:
            messagebox.showwarning("Device required", "Enter device name or MAC in the Device section.")
            return
        
        # Validate inputs based on mode
        mode = self.mode.get()
        
        if mode == "image":
            image = self.image_path.get().strip()
            if not image:
                messagebox.showwarning("Missing image", "Please select an image file.")
                return
            if not os.path.isfile(image):
                messagebox.showwarning("File not found", f"Image file does not exist:\n{image}")
                return
            self._append_log(f"\n--- Printing: {image} ---")
            self._send_command(f"image {image}")
        
        elif mode == "folder":
            folder = self.folder_path.get().strip()
            if not folder:
                messagebox.showwarning("Missing folder", "Please select a folder.")
                return
            if not os.path.isdir(folder):
                messagebox.showwarning("Directory not found", f"Folder does not exist:\n{folder}")
                return
            self._append_log(f"\n--- Printing folder: {folder} ---")
            self._send_command(f"folder {folder}")
        
        elif mode == "text":
            text_value = self.txt_text.get("1.0", "end-1c").strip()
            if not text_value:
                messagebox.showwarning("Missing text", "Please enter text to print.")
                return
            self._append_log(f"\n--- Printing text ---")
            self._send_command(f"text {text_value}")
        
        elif mode == "test":
            test_dir = TEST_IMAGE_FOLDER
            if not os.path.isdir(test_dir):
                messagebox.showwarning("Missing test folder", f"Could not find '{test_dir}/'. Create it or add some test images.")
                return
            self._append_log(f"\n--- Test print: {test_dir}/ ---")
            self._send_command(f"folder {test_dir}")
        
        elif mode == "feed":
            n = int(self.feed_lines.get())
            self._append_log(f"\n--- Feeding {n} lines ---")
            self._send_command(f"feed {n}")
        
        self._append_log(f"  [{mode.capitalize()} command sent]\n")
    
    def _pump_output(self):
        """Drain output queue and append to log."""
        try:
            while True:
                item = self.output_q.get_nowait()
                if item == "__PROCESS_DONE__":
                    continue
                self._append_log(item)
        except queue.Empty:
            pass
        self.after_id = self.root.after(50, self._pump_output)
    
    def _on_close(self):
        """Handle window close event."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self._append_log("\n[Cleaning up...]")
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
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
    app = MXW01GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
