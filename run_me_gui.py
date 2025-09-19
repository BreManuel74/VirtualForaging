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
        self.current_level_index = None
        # Add reward tracking
        self.reward_count = 0
        self.reward_callback = None
        self._reward_lock = threading.Lock()
    
    def _get_available_levels(self):
        """Get list of available level files sorted by their numeric order."""
        try:
            levels_path = os.path.join(os.getcwd(), self.levels_folder)
            # Get all JSON files
            level_files = [f for f in os.listdir(levels_path) if f.endswith('.json')]
            
            print(f"Found level files: {level_files}")  # Debug print
            
            # Sort levels based on their numeric value
            def extract_number(filename):
                # Extract number from level_X.json format
                try:
                    return int(''.join(filter(str.isdigit, filename)))
                except:
                    return float('inf')  # Put non-numeric levels at the end
            
            # Sort levels by their numeric value
            level_files.sort(key=extract_number)
            
            print(f"Sorted level files: {level_files}")  # Debug print
            
            # Store the ordered levels as a class attribute for later use
            self.ordered_levels = level_files
            
            return level_files
        except Exception as e:
            print(f"Error loading levels: {e}")  # Debug print
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
                            # Check for incoming data
                            self.client_socket.settimeout(1.0)
                            data = self.client_socket.recv(1024).decode('utf-8')
                            if data:
                                # Handle each line separately
                                for line in data.splitlines():
                                    if line:
                                        self._handle_received_data(line)
                            else:
                                # Empty data means connection closed
                                break
                        except socket.timeout:
                            # Timeout is normal, continue listening
                            continue
                        except Exception as e:
                            print(f"Error receiving data: {e}")
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
                
    def _handle_received_data(self, data):
        """Handle data received from the game client."""
        try:
            data = data.strip()
            if data.startswith("REWARD:"):
                with self._reward_lock:
                    self.reward_count += 1
                    if self.reward_callback:
                        self.reward_callback(self.reward_count)
        except Exception as e:
            print(f"Error handling received data: {e}")
            
    def set_reward_callback(self, callback):
        """Set a callback function to be called when a reward is received."""
        self.reward_callback = callback
        
    def reset_reward_count(self):
        """Reset the reward counter to zero."""
        with self._reward_lock:
            self.reward_count = 0
            if self.reward_callback:
                self.reward_callback(self.reward_count)
    
    def send_level_change(self, level_file):
        """Send level change command to the game."""
        command = f"CHANGE_LEVEL:{level_file}"
        if self.send_data(command):
            try:
                # Reset reward count for new level
                self.reset_reward_count()
                
                # Set the current index based on the actual level file
                if level_file in self.ordered_levels:
                    self.current_level_index = self.ordered_levels.index(level_file)
                    print(f"Set current level index to {self.current_level_index} for {level_file}")
                    print(f"Available levels: {self.ordered_levels}")  # Debug print
                return True
            except ValueError:
                print(f"Warning: Level {level_file} not found in ordered levels")
                return True
        return False
        
    def get_next_level(self):
        """Get the next level in the sequence."""
        if not self.ordered_levels or self.current_level_index is None:
            print(f"Debug - ordered_levels: {self.ordered_levels}")  # Debug print
            print(f"Debug - current_level_index: {self.current_level_index}")  # Debug print
            return None
            
        next_index = self.current_level_index + 1
        print(f"Debug - next_index: {next_index}, max index: {len(self.ordered_levels)-1}")  # Debug print
        
        if next_index >= len(self.ordered_levels):
            return None
            
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

    def set_initial_level(self, level_file):
        """Set the initial level index when starting a session."""
        if level_file in self.ordered_levels:
            self.current_level_index = self.ordered_levels.index(level_file)
            print(f"Set initial level index to {self.current_level_index} for {level_file}")
            return True
        return False

class MousePortalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Game Control")
        self.root.geometry("800x600")
        
        # For reward tracking
        self._last_reward_count = 0  # For internal tracking
        
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
        self.reward_count = tk.StringVar(value="0")
        
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
        control_frame.columnconfigure(2, weight=1)  # Make reward counter stick to right
        
        # Current Level Display
        ttk.Label(control_frame, text="Current Level:", style="Custom.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(control_frame, textvariable=self.current_level, style="Custom.TLabel").grid(row=0, column=1, sticky=tk.W)
        
        # Reward Counter Display
        reward_frame = ttk.Frame(control_frame)
        reward_frame.grid(row=0, column=2, padx=20, sticky=tk.E)
        ttk.Label(reward_frame, text="Rewards:", style="Custom.TLabel").pack(side=tk.LEFT)
        ttk.Label(reward_frame, textvariable=self.reward_count, style="Custom.TLabel").pack(side=tk.LEFT, padx=(5, 0))
        
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
        self.console = ScrolledText(console_frame, height=10, width=70, bg='black', fg='white', insertbackground='white')
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
            else:
                self.log_to_console("No more levels available")
    
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
                # Signal thorcam to stop - create flag in the same directory as run_me.py
                stop_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stop_recording.flag")
                
                # Check for any existing stop file and remove it first
                if os.path.exists(stop_file):
                    try:
                        os.remove(stop_file)
                    except:
                        pass
                
                # Create fresh stop file to signal thorcam
                with open(stop_file, 'w') as f:
                    f.write('stop')
                self.log_to_console("Signaling thorcam to stop...")
                
                # Wait for thorcam to detect flag and stop recording (increased timeout)
                max_wait = 15  # Maximum seconds to wait
                start_time = time.time()
                
                # Wait for evidence of thorcam stopping
                thorcam_stopped = False
                while time.time() - start_time < max_wait:
                    # Check for recently created video or log files
                    try:
                        output_dir = self.output_dir.get()
                        # Check for recently created video or log files
                        recent_files = [f for f in os.listdir(output_dir) 
                                      if (f.endswith('pupil_cam.avi') or f.endswith('_frame_log.txt'))
                                      and os.path.getmtime(os.path.join(output_dir, f)) > start_time]
                        if recent_files:
                            thorcam_stopped = True
                            self.log_to_console("Thorcam stopped successfully - output files detected")
                            break
                    except:
                        pass
                    time.sleep(0.5)
                        
                if not thorcam_stopped:
                    self.log_to_console("Warning: Thorcam may still be running - no confirmation of shutdown")
                
                # Always clean up the stop file regardless of detection status
                try:
                    if os.path.exists(stop_file):
                        os.remove(stop_file)
                        self.log_to_console("Removed stop recording flag")
                except Exception as e:
                    self.log_to_console(f"Warning: Could not remove stop recording flag: {str(e)}")
                
                # Cleanup resources in the correct order
                # 1. Stop TCP server first to prevent new commands
                if self.tcp_server:
                    self.log_to_console("Stopping TCP server...")
                    self.tcp_server.stop_server()
                    self.tcp_server = None
                    self.log_to_console("TCP server stopped")
                
                # 2. Stop the Panda3D process
                if self.process:
                    try:
                        self.log_to_console("Stopping Panda3D process...")
                        # Get the return code if process already finished
                        if self.process.poll() is not None:
                            exit_code = self.process.returncode
                            self.log_to_console(f"Panda3D process had already exited with code {exit_code}")
                        else:
                            # Gracefully terminate first
                            self.process.terminate()
                            try:
                                exit_code = self.process.wait(timeout=3)  # Give it 3 seconds to terminate gracefully
                                self.log_to_console(f"Panda3D process stopped gracefully (exit code: {exit_code})")
                            except subprocess.TimeoutExpired:
                                self.log_to_console("Forcing Panda3D process to stop...")
                                # Force kill if it doesn't terminate
                                self.process.kill()
                                exit_code = self.process.wait(timeout=2)
                                self.log_to_console(f"Panda3D process force-stopped (exit code: {exit_code})")
                    except Exception as e:
                        self.log_to_console(f"Error stopping Panda3D process: {str(e)}")
                    finally:
                        self.process = None
                
                # Let thorcam handle the stop file cleanup
                self.log_to_console("Session stopped successfully")
                
                # Clear current level display
                self.current_level.set("")
                
            except Exception as e:
                self.log_to_console(f"Error during shutdown: {str(e)}")
                # Still try to cleanup even if there was an error
                self.cleanup_resources()
    
    def _update_reward_display(self, count):
        """Callback function to update the reward display when new rewards are received."""
        self.reward_count.set(str(count))
    
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
            
            # Set up reward counting callback
            self.tcp_server.set_reward_callback(self._update_reward_display)
            
            # Set the initial level index
            initial_level = self.level_combobox.get()
            if not self.tcp_server.set_initial_level(initial_level):
                raise Exception(f"Invalid starting level: {initial_level}")
            
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
            
            # Initialize reward counting for the first level at 0
            self.tcp_server.reset_reward_count()
            
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