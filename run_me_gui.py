import os
import subprocess
import sys
import csv
import socket
import threading
import time
import signal
import atexit
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

class TCPDataServer:
    """
    TCP server to send data to the Panda3D game application.
    Thread-safe with proper cleanup mechanisms.
    """
    def __init__(self, host='localhost', port=0, levels_folder='Levels'):
        self.host = host
        self.port = port
        self.levels_folder = levels_folder
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.server_thread = None
        self.available_levels = self._get_available_levels()
        self._shutdown_lock = threading.Lock()
        self._cleanup_registered = False
        self.current_level_index = 0  # Track the current level index
        
    def _get_available_levels(self):
        """Get list of available level files sorted by their numeric order."""
        try:
            levels_path = os.path.join(os.getcwd(), self.levels_folder)
            # Get all JSON files
            level_files = [f for f in os.listdir(levels_path) if f.endswith('.json')]
            
            # Sort levels based on their numeric value
            def extract_number(filename):
                # Extract number from level_X.json format
                try:
                    return int(''.join(filter(str.isdigit, filename)))
                except:
                    return float('inf')  # Put non-numeric levels at the end
            
            # Sort levels by their numeric value
            level_files.sort(key=extract_number)
            
            # Store the ordered levels as a class attribute for later use
            self.ordered_levels = level_files
            
            return level_files
        except:
            self.ordered_levels = []
            return []
    
    def list_available_levels(self):
        """Return formatted list of available levels in sequential order."""
        if not self.available_levels:
            return "No level files found."
        
        level_list = "Available levels (in sequential order):\n"
        for idx, level in enumerate(self.ordered_levels, 1):
            level_list += f"{idx}: {level}\n"
        return level_list.strip()
            
    def get_ordered_levels(self):
        """Get the list of levels in their sequential numeric order."""
        return self.ordered_levels.copy()
        
    def start_server(self):
        """Start the TCP server in a separate thread."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            # Get the actual port number if we used 0 (auto-assign)
            self.port = self.server_socket.getsockname()[1]
            print(f"TCP server started on {self.host}:{self.port}")
            
            self.running = True
            
            # Register cleanup on exit if not already done
            if not self._cleanup_registered:
                atexit.register(self.stop_server)
                self._cleanup_registered = True
            
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True, name="TCPServer")
            self.server_thread.start()
            
            return self.port
        except Exception as e:
            print(f"Failed to start TCP server: {e}")
            self.stop_server()
            raise
    
    def _server_loop(self):
        """Main server loop to accept connections with proper error handling."""
        try:
            self.server_socket.listen(1)
            print("Waiting for Panda3D game to connect...")
            
            while self.running:
                if not self.server_socket:
                    break
                    
                self.server_socket.settimeout(1.0)  # Non-blocking accept
                try:
                    self.client_socket, client_address = self.server_socket.accept()
                    print(f"Game connected from {client_address}")
                    
                    # Keep connection alive while running
                    while self.running and self.client_socket:
                        try:
                            # Check if connection is still alive
                            self.client_socket.settimeout(1.0)
                            time.sleep(0.1)
                        except:
                            break
                    break
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Socket accept error: {e}")
                    break
        except Exception as e:
            if self.running:
                print(f"Server loop error: {e}")
        finally:
            print("Server loop ended")
    
    def send_data(self, data):
        """Send data to the connected game client with thread safety."""
        with self._shutdown_lock:
            if not self.running:
                return False
                
            if self.client_socket:
                try:
                    message = f"{data}\n"
                    self.client_socket.send(message.encode('utf-8'))
                    return True
                except Exception as e:
                    print(f"Error sending data: {e}")
                    # Close broken connection
                    try:
                        self.client_socket.close()
                    except:
                        pass
                    self.client_socket = None
                    return False
            else:
                print("No client connected")
                return False
    
    def send_level_change(self, level_file):
        """Send level change command to the game."""
        command = f"CHANGE_LEVEL:{level_file}"
        # Update current_level_index if the level change is successful
        if self.send_data(command):
            try:
                self.current_level_index = self.ordered_levels.index(level_file)
                return True
            except ValueError:
                print(f"Warning: Level {level_file} not found in ordered levels")
                return True
        return False
        
    def get_next_level(self):
        """Get the next level in the sequence."""
        if not self.ordered_levels:
            return None
        next_index = (self.current_level_index + 1) % len(self.ordered_levels)
        return self.ordered_levels[next_index]
    
    def stop_server(self):
        """Stop the TCP server with proper thread cleanup."""
        with self._shutdown_lock:
            if not self.running:
                return
                
            print("Stopping TCP server...")
            self.running = False
            
            # Close client connection
            if self.client_socket:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
            
            # Close server socket
            if self.server_socket:
                try:
                    self.server_socket.close()
                except:
                    pass
                self.server_socket = None
            
            # Wait for server thread to finish
            if self.server_thread and self.server_thread.is_alive():
                print("Waiting for server thread to finish...")
                self.server_thread.join(timeout=3)
                if self.server_thread.is_alive():
                    print("Warning: Server thread did not stop cleanly")
                else:
                    print("Server thread stopped successfully")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.stop_server()

class MousePortalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Control")
        self.root.geometry("800x600")
        
        # Set dark theme
        self.root.configure(bg='black')
        style = ttk.Style()
        style.configure(".", background='black', foreground='white')
        style.configure("TFrame", background='black')
        
        # Configure custom styles for better visibility
        style.configure("Custom.TLabelframe", background='black')
        style.configure("Custom.TLabelframe.Label", background='black', foreground='white')
        style.configure("Custom.TLabel", background='black', foreground='white')
        
        # Default styles
        style.configure("TLabelframe", background='black')
        style.configure("TLabelframe.Label", background='black', foreground='white')
        style.configure("TLabel", background='black', foreground='white')
        
        # Configure button style with proper contrast
        style.configure("TButton", 
            background='gray30', 
            foreground='black',
            bordercolor='gray40',
            lightcolor='gray40',
            darkcolor='gray20'
        )
        style.map("TButton",
            background=[('active', 'gray40'), ('pressed', 'gray20')],
            foreground=[('active', 'black'), ('pressed', 'black')]
        )
        
        # Configure entry and combobox styles
        style.configure("TEntry", 
            fieldbackground='gray20', 
            foreground='black',
            insertcolor='black'
        )
        style.map("TEntry",
            fieldbackground=[('readonly', 'gray20')],
            foreground=[('readonly', 'black')]
        )
        style.configure("TCombobox", 
            fieldbackground='gray20', 
            background='gray30',
            foreground='black',
            arrowcolor='black',
            selectbackground='gray40',
            selectforeground='black'
        )
        style.map('TCombobox',
            fieldbackground=[('readonly', 'gray20')],
            selectbackground=[('readonly', 'gray40')],
            background=[('readonly', 'gray30'), ('active', 'gray40')],
            foreground=[('readonly', 'black')],
            arrowcolor=[('readonly', 'black')]
        )
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="10", style="TFrame")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup variables
        self.animal_name = tk.StringVar()
        self.batch_id = tk.StringVar()
        self.teensy_port = tk.StringVar(value="COM3")
        self.current_level = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd())
        
        # Setup GUI elements
        self.create_setup_frame(main_frame)
        self.create_level_control_frame(main_frame)
        self.create_console_frame(main_frame)
        
        # TCP Server and Process
        self.tcp_server = None
        self.process = None
        self.command_thread = None
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
    def create_setup_frame(self, parent):
        # Setup Frame
        setup_frame = ttk.LabelFrame(parent, text="Setup", padding="5", style="Custom.TLabelframe")
        setup_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Animal Name
        ttk.Label(setup_frame, text="Animal Name:", style="Custom.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(setup_frame, textvariable=self.animal_name, style="TEntry").grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Batch ID
        ttk.Label(setup_frame, text="Batch ID:", style="Custom.TLabel").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(setup_frame, textvariable=self.batch_id, style="TEntry").grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Teensy Port
        ttk.Label(setup_frame, text="Teensy Port:", style="Custom.TLabel").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(setup_frame, textvariable=self.teensy_port, style="TEntry").grid(row=2, column=1, sticky=(tk.W, tk.E))
        
        # Output Directory
        ttk.Label(setup_frame, text="Output Directory:", style="Custom.TLabel").grid(row=3, column=0, sticky=tk.W)
        dir_frame = ttk.Frame(setup_frame)
        dir_frame.grid(row=3, column=1, sticky=(tk.W, tk.E))
        ttk.Entry(dir_frame, textvariable=self.output_dir, style="TEntry").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.RIGHT)
        
        # Level Selection
        ttk.Label(setup_frame, text="Starting Level:", style="Custom.TLabel").grid(row=4, column=0, sticky=tk.W)
        self.level_combobox = ttk.Combobox(setup_frame, state="readonly")
        self.level_combobox.grid(row=4, column=1, sticky=(tk.W, tk.E))
        self.update_level_list()
        
        # Start Button
        ttk.Button(setup_frame, text="Start Session", command=self.start_session).grid(row=5, column=0, columnspan=2, pady=10)
        
        # Configure grid
        setup_frame.columnconfigure(1, weight=1)
        
    def create_level_control_frame(self, parent):
        # Level Control Frame
        control_frame = ttk.LabelFrame(parent, text="Level Control", padding="5", style="Custom.TLabelframe")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Current Level Display
        ttk.Label(control_frame, text="Current Level:", style="Custom.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.current_level, style="Custom.TLabel").grid(row=0, column=1, sticky=tk.W)
        
        # Level Control Buttons
        button_frame = ttk.Frame(control_frame, style="TFrame")
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(button_frame, text="Next Level", command=self.change_to_next_level).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="List Levels", command=self.show_level_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop Session", command=self.stop_session).pack(side=tk.LEFT, padx=5)
        
    def create_console_frame(self, parent):
        # Console Frame
        console_frame = ttk.LabelFrame(parent, text="Console Output", padding="5", style="Custom.TLabelframe")
        console_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Console Output with dark theme
        self.console = ScrolledText(console_frame, height=10, width=70, bg='black', fg='black', insertbackground='white')
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Command Entry
        cmd_frame = ttk.Frame(console_frame, style="TFrame")
        cmd_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(cmd_frame, text="Command:", style="Custom.TLabel").pack(side=tk.LEFT)
        self.cmd_entry = ttk.Entry(cmd_frame)
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(cmd_frame, text="Send", command=self.send_command).pack(side=tk.LEFT)
        
        # Bind Enter key to send command
        self.cmd_entry.bind('<Return>', lambda e: self.send_command())
        
    def update_level_list(self):
        try:
            levels_path = os.path.join(os.getcwd(), 'Levels')
            level_files = [f for f in os.listdir(levels_path) if f.endswith('.json')]
            
            # Sort levels based on their numeric value
            def extract_number(filename):
                try:
                    return int(''.join(filter(str.isdigit, filename)))
                except:
                    return float('inf')
            
            level_files.sort(key=extract_number)
            self.level_combobox['values'] = level_files
            if level_files:
                self.level_combobox.set(level_files[0])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load level files: {str(e)}")
    
    def browse_output_dir(self):
        from tkinter import filedialog
        dir_path = filedialog.askdirectory(initialdir=self.output_dir.get())
        if dir_path:
            self.output_dir.set(dir_path)
    
    def log_to_console(self, message):
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
    
    def send_command(self):
        command = self.cmd_entry.get().strip()
        if command and self.tcp_server:
            success = self.tcp_server.send_data(command)
            if success:
                self.log_to_console(f"Sent: {command}")
            else:
                self.log_to_console("Failed to send command")
        self.cmd_entry.delete(0, tk.END)
    
    def change_to_next_level(self):
        if self.tcp_server:
            next_level = self.tcp_server.get_next_level()
            if next_level:
                success = self.tcp_server.send_level_change(next_level)
                if success:
                    self.current_level.set(next_level)
                    current_idx = self.tcp_server.current_level_index + 1
                    total_levels = len(self.tcp_server.ordered_levels)
                    self.log_to_console(f"Advanced to: {next_level} (Level {current_idx} of {total_levels})")
                else:
                    self.log_to_console("Failed to change level")
    
    def show_level_list(self):
        if self.tcp_server:
            self.log_to_console("\n" + self.tcp_server.list_available_levels())
    
    def log_run(self, animal_name, level_file, phase_file, batch_id):
        try:
            log_dir = "Progress_Reports"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{animal_name}_log.csv")
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            level_file_name = os.path.basename(level_file)
            phase_file_name = os.path.basename(phase_file)
            
            write_header = not os.path.exists(log_file)
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["Date", "Time", "Level File", "Phase File", "Batch ID"])
                writer.writerow([date_str, time_str, level_file_name, phase_file_name, batch_id])
        except Exception as e:
            self.log_to_console(f"Error logging run: {str(e)}")
    
    def cleanup_resources(self):
        if self.tcp_server:
            self.tcp_server.stop_server()
            self.tcp_server = None
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
            self.process = None
    
    def kill_process_tree(self, pid):
        """Kill a process and all its children."""
        try:
            import psutil
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # First try to terminate children gracefully
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
            
            # Try to terminate parent gracefully
            parent.terminate()
            
            # Wait for processes to terminate
            _, alive = psutil.wait_procs(children + [parent], timeout=3)
            
            # If any processes are still alive, kill them forcefully
            for p in alive:
                try:
                    p.kill()
                except:
                    pass
                    
        except Exception as e:
            self.log_to_console(f"Error killing process tree: {str(e)}")

    def stop_session(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the current session?"):
            self.log_to_console("\nShutting down...")
            
            try:
                # First create the stop file for thorcam
                stop_file = os.path.join(self.output_dir.get(), "stop_recording.flag")
                with open(stop_file, 'w') as f:
                    f.write('stop')
                self.log_to_console("Signaling thorcam to stop...")
                
                # Give thorcam a moment to detect the flag and cleanup
                time.sleep(2)
                
                # Cleanup resources in the correct order
                if self.tcp_server:
                    # First stop the TCP server
                    self.tcp_server.stop_server()
                    self.tcp_server = None
                
                if self.process:
                    try:
                        # Gracefully terminate the Panda3D process
                        self.process.terminate()
                        # Wait for process to end
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # Force kill if process doesn't terminate
                            self.process.kill()
                            self.process.wait()
                    except Exception as e:
                        self.log_to_console(f"Error stopping process: {str(e)}")
                    finally:
                        self.process = None
                
                # Remove the stop file
                try:
                    if os.path.exists(stop_file):
                        os.remove(stop_file)
                except Exception as e:
                    self.log_to_console(f"Warning: Could not remove stop file: {str(e)}")
                
                self.log_to_console("Session stopped successfully")
                
                # Clear current level display
                self.current_level.set("")
                
            except Exception as e:
                self.log_to_console(f"Error during shutdown: {str(e)}")
                # Still try to cleanup even if there was an error
                self.cleanup_resources()
    
    def start_session(self):
        # Validate inputs
        if not all([self.animal_name.get(), self.batch_id.get(), self.teensy_port.get()]):
            messagebox.showerror("Error", "Please fill in all required fields")
            return
        
        # Get file paths
        level_file = os.path.join(os.getcwd(), 'Levels', self.level_combobox.get())
        phase_file = os.path.join(os.getcwd(), 'Phases', 'final.py')
        
        # Log the run
        self.log_run(self.animal_name.get(), level_file, phase_file, self.batch_id.get())
        
        # Convert level file path to relative with forward slashes
        level_file_path = level_file.replace(os.getcwd() + os.sep, '').replace('\\', '/')
        
        try:
            # Start TCP server
            self.tcp_server = TCPDataServer()
            server_port = self.tcp_server.start_server()
            
            # Set up environment variables
            env = os.environ.copy()
            env["LEVEL_CONFIG_PATH"] = level_file_path
            env["TCP_SERVER_PORT"] = str(server_port)
            env["OUTPUT_DIR"] = self.output_dir.get()
            env["BATCH_ID"] = self.batch_id.get()
            env["TEENSY_PORT"] = self.teensy_port.get()
            
            # Start the Panda3D game as a subprocess
            self.process = subprocess.Popen([sys.executable, phase_file], env=env)
            
            # Set initial current level
            self.current_level.set(self.level_combobox.get())
            
            # Log success
            self.log_to_console(f"Started session with:")
            self.log_to_console(f"Animal: {self.animal_name.get()}")
            self.log_to_console(f"Level: {self.level_combobox.get()}")
            self.log_to_console(f"Batch ID: {self.batch_id.get()}")
            self.log_to_console(f"Teensy Port: {self.teensy_port.get()}")
            self.log_to_console(f"TCP server started on port {server_port}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start session: {str(e)}")
            self.cleanup_resources()

def main():
    root = tk.Tk()
    app = MousePortalGUI(root)
    
    def on_closing():
        if app.process or app.tcp_server:
            # Use stop_session to handle cleanup, but don't show confirmation dialog again
            app.stop_session()
            root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()