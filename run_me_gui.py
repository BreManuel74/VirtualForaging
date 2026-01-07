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

# Import psutil at module level to avoid import overhead later
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, process tree killing will be limited")

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
        self._client_lock = threading.Lock()  # Lock for client socket access
        self._running_lock = threading.Lock() # Lock for running flag access
        self._cleanup_registered = False
        self.current_level_index = None
        # Add reward tracking
        self.reward_count = 0
        self.reward_callback = None
        self._reward_lock = threading.Lock()
        # Add socket timeout
        self._socket_timeout = 1.0
        # Cache for reward thresholds to avoid repeated file I/O
        self._threshold_cache = {}

    def _get_level_reward_threshold(self, level_file):
        """Read the reward threshold from a level file (cached for performance)."""
        # Check cache first to avoid repeated file I/O
        if level_file in self._threshold_cache:
            return self._threshold_cache[level_file]
        
        try:
            import json
            level_path = os.path.join(os.getcwd(), 'Levels', level_file)
            with open(level_path, 'r') as f:
                level_data = json.load(f)
                threshold = level_data.get('reward_threshold', float('inf'))
                #print(f"Read reward threshold {threshold} from {level_file}")
                if not isinstance(threshold, (int, float)) or threshold <= 0:
                    print(f"Invalid threshold value {threshold}, using infinity")
                    threshold = float('inf')
                
                # Cache the result for future use
                self._threshold_cache[level_file] = threshold
                return threshold
        except Exception as e:
            print(f"Error reading reward threshold from {level_file}: {e}")
            threshold = float('inf')
            self._threshold_cache[level_file] = threshold
            return threshold
        
    def _get_available_levels(self):
        """Get list of available level files sorted by their numeric order."""
        try:
            levels_path = os.path.join(os.getcwd(), self.levels_folder)
            # Get all JSON files
            level_files = [f for f in os.listdir(levels_path) if f.endswith('.json')]
            
            #print(f"Found level files: {level_files}")  # Debug print
            
            # Sort levels based on their numeric value
            def extract_number(filename):
                # Extract number from level_X.json format
                try:
                    return int(''.join(filter(str.isdigit, filename)))
                except:
                    return float('inf')  # Put non-numeric levels at the end
            
            # Sort levels by their numeric value
            level_files.sort(key=extract_number)
            
            #print(f"Sorted level files: {level_files}")  # Debug print
            
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
            if not self.server_socket:
                return
                
            self.server_socket.listen(1)
            print("Waiting for Panda3D game to connect...")
            
            while True:
                # Check running state with proper locking
                with self._running_lock:
                    if not self.running:
                        break
                    
                if not self.server_socket:
                    break
                    
                try:
                    self.server_socket.settimeout(self._socket_timeout)
                    client_socket, client_address = self.server_socket.accept()
                    
                    # Set client socket with proper locking
                    with self._client_lock:
                        self.client_socket = client_socket
                        print(f"Game connected from {client_address}")
                    
                    # Process data while connection is active
                    while True:
                        # Check running state again
                        with self._running_lock:
                            if not self.running:
                                break
                                
                        # Verify client socket still exists
                        with self._client_lock:
                            if not self.client_socket:
                                break
                            
                            try:
                                self.client_socket.settimeout(self._socket_timeout)
                                data = self.client_socket.recv(1024).decode('utf-8')
                                
                                if data:
                                    # Handle each line separately with reward lock protection
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
                    
                    # Clean up client socket after loop ends
                    self._close_client_socket()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    with self._running_lock:
                        if self.running:
                            print(f"Socket accept error: {e}")
                    break
                    
        except Exception as e:
            with self._running_lock:
                if self.running:
                    print(f"Server loop error: {e}")
        finally:
            print("Server loop ended")
            # Ensure client socket is closed
            self._close_client_socket()
    
    def send_data(self, data):
        """Send data to the connected game client with thread safety (non-blocking)."""
        # First check running state
        with self._running_lock:
            if not self.running:
                return False

        # Schedule send on background thread to never block GUI
        def _async_send():
            with self._client_lock:
                if not self.client_socket:
                    print("No client socket available for send")
                    return False
                    
                try:
                    message = f"{data}\n"
                    # Set a very short timeout - if client can't receive, fail fast
                    self.client_socket.settimeout(0.1)  # 100ms max wait
                    
                    send_start = time.time()
                    self.client_socket.sendall(message.encode('utf-8'))
                    send_time = time.time() - send_start
                    
                    if send_time > 0.05:  # Warn if send takes >50ms
                        print(f"WARNING: Socket send took {send_time*1000:.1f}ms - client may be slow")
                    
                    return True
                except socket.timeout:
                    print(f"ERROR: Socket send timeout after 100ms - client buffer full or not reading")
                    return False
                except Exception as e:
                    print(f"Error sending data: {e}")
                    return False
        
        # For non-blocking operation, send on background thread
        threading.Thread(target=_async_send, daemon=True, name="SocketSender").start()
        return True  # Return immediately, actual send happens in background
                
    def _close_client_socket(self):
        """Helper method to safely close client socket with proper locking."""
        with self._client_lock:
            if self.client_socket:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
                
    def _handle_received_data(self, data):
        """Handle data received from the game client."""
        try:
            data = data.strip()
            if data.startswith("REWARD:"):
                with self._reward_lock:
                    self.reward_count += 1
                    count = self.reward_count  # Capture current count
                    threshold_reached = False
                    
                    if hasattr(self, 'current_reward_threshold'):
                        threshold_reached = self.reward_count >= self.current_reward_threshold
                        # Only print debug info when threshold is reached or every 10 rewards
                        if threshold_reached or self.reward_count % 10 == 0:
                            print(f"Reward check: count={self.reward_count}, threshold={self.current_reward_threshold}, reached={threshold_reached}")
                    
                    # Schedule callback on main thread - don't block the TCP thread!
                    if self.reward_callback:
                        # Use a copy of the values to avoid race conditions
                        self._schedule_callback(count, threshold_reached)
        except Exception as e:
            print(f"Error handling received data: {e}")
    
    def _schedule_callback(self, count, threshold_reached):
        """Schedule the reward callback to run on the main thread (non-blocking)."""
        # This method can be overridden by the GUI to use proper thread scheduling
        if self.reward_callback:
            self.reward_callback(count, threshold_reached)
            
    def set_reward_callback(self, callback):
        """Set a callback function to be called when a reward is received."""
        self.reward_callback = callback
        
    def reset_reward_count(self):
        """Reset the reward counter to zero."""
        with self._reward_lock:
            self.reward_count = 0
            # Don't call callback here - let the GUI update happen naturally
            # The callback will be triggered by the next reward event
    
    def send_level_change(self, level_file):
        """Send level change command to the game."""
        command = f"CHANGE_LEVEL:{level_file}"
        if self.send_data(command):
            try:
                # Note: reward count should be reset BEFORE calling this method
                # to prevent race conditions with incoming rewards
                
                # Set the current index based on the actual level file
                if level_file in self.ordered_levels:
                    self.current_level_index = self.ordered_levels.index(level_file)
                    #print(f"Set current level index to {self.current_level_index} for {level_file}")
                    
                    # Get and store the reward threshold for the new level
                    try:
                        self.current_reward_threshold = self._get_level_reward_threshold(level_file)
                        #print(f"Successfully loaded reward threshold: {self.current_reward_threshold}")
                    except Exception as e:
                        print(f"Error loading reward threshold: {e}")
                        self.current_reward_threshold = float('inf')
                    
                    return True
            except ValueError:
                print(f"Warning: Level {level_file} not found in ordered levels")
                return True
        return False
        
    def get_next_level(self):
        """Get the next level in the sequence."""
        if not self.ordered_levels or self.current_level_index is None:
            #print(f"Debug - ordered_levels: {self.ordered_levels}")  # Debug print
            #print(f"Debug - current_level_index: {self.current_level_index}")  # Debug print
            return None
            
        next_index = self.current_level_index + 1
        #print(f"Debug - next_index: {next_index}, max index: {len(self.ordered_levels)-1}")  # Debug print
        
        if next_index >= len(self.ordered_levels):
            return None
            
        return self.ordered_levels[next_index]
        
    def get_previous_level(self):
        """Get the previous level in the sequence."""
        if not self.ordered_levels or self.current_level_index is None:
            return None
            
        prev_index = self.current_level_index - 1
        
        if prev_index < 0:
            return None
            
        return self.ordered_levels[prev_index]

    def stop_server(self):
        """Stop the TCP server with proper thread cleanup."""
        print("Stopping TCP server...")
        
        # First set running to false with proper locking
        with self._running_lock:
            if not self.running:
                return
            self.running = False
        
        # Then acquire shutdown lock for the entire cleanup sequence
        with self._shutdown_lock:
            # Close client connection with proper locking
            self._close_client_socket()
            
            # Close server socket
            if self.server_socket:
                try:
                    self.server_socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    self.server_socket.close()
                except:
                    pass
                self.server_socket = None
            
            # Wait for server thread to finish with timeout
            if self.server_thread and self.server_thread.is_alive():
                print("Waiting for server thread to finish...")
                try:
                    self.server_thread.join(timeout=5)  # Increased timeout
                    if self.server_thread.is_alive():
                        print("Warning: Server thread did not stop cleanly")
                    else:
                        print("Server thread stopped successfully")
                except Exception as e:
                    print(f"Error during thread cleanup: {e}")
            
            # Clear thread reference
            self.server_thread = None
            
            print("TCP server stopped")
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.stop_server()

    def set_initial_level(self, level_file):
        """Set the initial level index when starting a session."""
        if level_file in self.ordered_levels:
            self.current_level_index = self.ordered_levels.index(level_file)
            #print(f"Set initial level index to {self.current_level_index} for {level_file}")
            
            # Set initial reward threshold
            try:
                self.current_reward_threshold = self._get_level_reward_threshold(level_file)
                #print(f"Initial reward threshold set to: {self.current_reward_threshold}")
            except Exception as e:
                print(f"Error setting initial reward threshold: {e}")
                self.current_reward_threshold = float('inf')
            
            return True
        return False

class KaufmanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kaufman Game Control")
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
        self.session_active = False  # Track if a session is active
        self.selected_progress_report = tk.StringVar()  # Track selected progress report
        
        # Initialize list for session-sensitive widgets before creating frames
        self.session_sensitive_widgets = []
        
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
        animal_entry = ttk.Entry(setup_frame, textvariable=self.animal_name, style="TEntry")
        animal_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.session_sensitive_widgets.append(animal_entry)
        
        # Batch ID
        ttk.Label(setup_frame, text="Batch ID:", style="Custom.TLabel").grid(row=1, column=0, sticky=tk.W)
        batch_entry = ttk.Entry(setup_frame, textvariable=self.batch_id, style="TEntry")
        batch_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.session_sensitive_widgets.append(batch_entry)
        
        # Teensy Port
        ttk.Label(setup_frame, text="Teensy Port:", style="Custom.TLabel").grid(row=2, column=0, sticky=tk.W)
        teensy_entry = ttk.Entry(setup_frame, textvariable=self.teensy_port, style="TEntry")
        teensy_entry.grid(row=2, column=1, sticky=(tk.W, tk.E))
        self.session_sensitive_widgets.append(teensy_entry)
        
        # Output Directory
        ttk.Label(setup_frame, text="Output Directory:", style="Custom.TLabel").grid(row=3, column=0, sticky=tk.W)
        dir_frame = ttk.Frame(setup_frame)
        dir_frame.grid(row=3, column=1, sticky=(tk.W, tk.E))
        output_entry = ttk.Entry(dir_frame, textvariable=self.output_dir, style="TEntry")
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_output_dir)
        browse_btn.pack(side=tk.RIGHT)
        self.session_sensitive_widgets.extend([output_entry, browse_btn])
        
        # Progress Report Selection
        ttk.Label(setup_frame, text="Progress Report:", style="Custom.TLabel").grid(row=4, column=0, sticky=tk.W)
        self.progress_report_combobox = ttk.Combobox(setup_frame, textvariable=self.selected_progress_report, state="readonly")
        self.progress_report_combobox.grid(row=4, column=1, sticky=(tk.W, tk.E))
        self.update_progress_report_list()
        self.session_sensitive_widgets.append(self.progress_report_combobox)
        
        # Start Button
        self.start_button = ttk.Button(setup_frame, text="Start Session", command=self.start_session)
        self.start_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.session_sensitive_widgets.append(self.start_button)
        
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
        
        # Create buttons and store references for state management
        prev_level_btn = ttk.Button(button_frame, text="Previous Level", command=self.change_to_previous_level)
        prev_level_btn.pack(side=tk.LEFT, padx=5)
        next_level_btn = ttk.Button(button_frame, text="Next Level", command=self.change_to_next_level)
        next_level_btn.pack(side=tk.LEFT, padx=5)
        list_levels_btn = ttk.Button(button_frame, text="List Levels", command=self.show_level_list)
        list_levels_btn.pack(side=tk.LEFT, padx=5)
        stop_session_btn = ttk.Button(button_frame, text="Stop Session", command=self.stop_session)
        stop_session_btn.pack(side=tk.LEFT, padx=5)
        
        # Add the previous level button to session sensitive widgets
        prev_level_btn.is_session_control = True
        
        # Add the level control buttons to session sensitive widgets with opposite behavior
        # These buttons should be enabled during a session and disabled when no session is active
        next_level_btn.is_session_control = True
        list_levels_btn.is_session_control = True
        stop_session_btn.is_session_control = True
        self.session_sensitive_widgets.extend([next_level_btn, list_levels_btn, stop_session_btn])
        
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
        
    def update_progress_report_list(self):
        """Update the list of available progress reports."""
        try:
            log_dir = "Progress_Reports"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                self.progress_report_combobox['values'] = []
                return
            
            # Get all CSV files in the Progress_Reports directory
            report_files = [f for f in os.listdir(log_dir) if f.endswith('_log.csv')]
            
            if report_files:
                self.progress_report_combobox['values'] = report_files
                self.progress_report_combobox.set(report_files[0])
            else:
                self.progress_report_combobox['values'] = []
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load progress reports: {str(e)}")
    
    def get_starting_level_from_report(self, report_file):
        """
        Get the starting level and reward count from the most recent entry in the progress report.
        If the report doesn't exist or is empty, return (level_1.json, 0).
        Returns: tuple (level_name, reward_count)
        """
        try:
            log_dir = "Progress_Reports"
            report_path = os.path.join(log_dir, report_file)
            
            # If report doesn't exist, return level_1.json with 0 rewards
            if not os.path.exists(report_path):
                self.log_to_console(f"Progress report not found, starting with level_1.json")
                return ("level_1.json", 0)
            
            # Read the CSV file
            with open(report_path, mode="r", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Check if file is empty or only has header
                if len(rows) <= 1:
                    self.log_to_console(f"Progress report is empty, starting with level_1.json")
                    return ("level_1.json", 0)
                
                # Get the last entry (most recent)
                last_row = rows[-1]
                
                # Extract level from the last row (column index 2)
                if len(last_row) >= 3:
                    level_entry = last_row[2]
                    reward_count = 0  # Default to 0
                    
                    # Check if this is a "Session End" entry
                    if "(Session End" in level_entry:
                        # Extract the level name before " (Session End"
                        level_name = level_entry.split(" (Session End")[0].strip()
                        
                        # Try to extract reward count from the session end entry
                        try:
                            # Format is "level_X.json (Session End - N rewards)"
                            reward_part = level_entry.split("Session End - ")[1]
                            reward_count_str = reward_part.split(" rewards")[0].strip()
                            reward_count = int(reward_count_str)
                            self.log_to_console(f"Found reward count from session end: {reward_count}")
                        except (IndexError, ValueError) as e:
                            self.log_to_console(f"Could not parse reward count, defaulting to 0")
                            reward_count = 0
                    else:
                        level_name = level_entry.strip()
                        # For non-session-end entries, default to 0 rewards
                        reward_count = 0
                    
                    # Validate that the level file exists
                    level_path = os.path.join(os.getcwd(), 'Levels', level_name)
                    if os.path.exists(level_path):
                        self.log_to_console(f"Resuming from last level: {level_name} with {reward_count} rewards")
                        return (level_name, reward_count)
                    else:
                        self.log_to_console(f"Level {level_name} not found, starting with level_1.json")
                        return ("level_1.json", 0)
                else:
                    self.log_to_console(f"Invalid progress report format, starting with level_1.json")
                    return ("level_1.json", 0)
                    
        except Exception as e:
            self.log_to_console(f"Error reading progress report: {str(e)}, starting with level_1.json")
            return ("level_1.json", 0)
        
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
        """Log a message to the console with optimal performance."""
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        # Don't force updates - let Tkinter handle it naturally during idle time
        # This prevents blocking the UI thread
    
    def send_command(self):
        command = self.cmd_entry.get().strip()
        if command and self.tcp_server:
            success = self.tcp_server.send_data(command)
            if success:
                self.log_to_console(f"Sent: {command}")
            else:
                self.log_to_console("Failed to send command")
        self.cmd_entry.delete(0, tk.END)
    
    def log_level_change(self, level_file):
        """Log a level change to the progress report CSV asynchronously."""
        # Schedule file I/O on a background thread to avoid blocking GUI
        def _write_log():
            try:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                
                if hasattr(self, 'log_file'):
                    with open(self.log_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([date_str, time_str, level_file, self.batch_id.get()])
            except Exception as e:
                # Schedule error logging back on main thread
                self.root.after(0, lambda: self.log_to_console(f"Error logging level change: {str(e)}"))
        
        # Execute file I/O on background thread
        threading.Thread(target=_write_log, daemon=True, name="LogWriter").start()

    def change_to_next_level(self):
        if self.tcp_server:
            next_level = self.tcp_server.get_next_level()
            if next_level:
                # Reset reward count FIRST before any UI updates to prevent race conditions
                self.tcp_server.reset_reward_count()
                
                # Update UI for instant feedback
                start_time = time.time()
                self.current_level.set(next_level)
                self.reward_count.set("0")
                ui_time = time.time() - start_time
                
                print(f"Level change timing: UI={ui_time*1000:.1f}ms")
                
                # Do the network/file operations asynchronously to avoid blocking GUI
                def _async_level_change():
                    net_start = time.time()
                    success = self.tcp_server.send_level_change(next_level)
                    net_time = time.time() - net_start
                    
                    print(f"Network operation completed: {net_time*1000:.1f}ms")
                    
                    # Schedule UI updates back on main thread
                    if success:
                        current_idx = self.tcp_server.current_level_index + 1
                        total_levels = len(self.tcp_server.ordered_levels)
                        self.root.after(0, lambda: self.log_to_console(f"Advanced to: {next_level} (Level {current_idx} of {total_levels})"))
                        self.log_level_change(next_level)
                    else:
                        self.root.after(0, lambda: self.log_to_console("Failed to change level"))
                
                # Execute network operation on background thread - don't block GUI!
                threading.Thread(target=_async_level_change, daemon=True, name="LevelChanger").start()
            else:
                self.log_to_console("No more levels available")
    
    def change_to_previous_level(self):
        """Change to the previous level in the sequence."""
        if self.tcp_server:
            prev_level = self.tcp_server.get_previous_level()
            if prev_level:
                # Update UI first for instant feedback
                self.current_level.set(prev_level)
                self.reward_count.set("0")
                
                # Do the network/file operations asynchronously to avoid blocking GUI
                def _async_level_change():
                    success = self.tcp_server.send_level_change(prev_level)
                    
                    # Schedule UI updates back on main thread
                    if success:
                        current_idx = self.tcp_server.current_level_index + 1
                        total_levels = len(self.tcp_server.ordered_levels)
                        self.root.after(0, lambda: self.log_to_console(f"Changed to previous level: {prev_level} (Level {current_idx} of {total_levels})"))
                        self.log_level_change(prev_level)
                    else:
                        self.root.after(0, lambda: self.log_to_console("Failed to change level"))
                
                # Execute network operation on background thread - don't block GUI!
                threading.Thread(target=_async_level_change, daemon=True, name="LevelChanger").start()
            else:
                self.log_to_console("Already at first level")

    def show_level_list(self):
        if self.tcp_server:
            self.log_to_console("\n" + self.tcp_server.list_available_levels())
    
    def log_run(self, animal_name, level_file, batch_id):
        try:
            log_dir = "Progress_Reports"
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"{animal_name}_log.csv")
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            
            level_file_name = os.path.basename(level_file)
            
            write_header = not os.path.exists(self.log_file)
            with open(self.log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["Date", "Time", "Level", "Batch ID"])
                writer.writerow([date_str, time_str, level_file_name, batch_id])
        except Exception as e:
            self.log_to_console(f"Error logging run: {str(e)}")
    
    def _set_session_state(self, active: bool):
        """Enable or disable widgets based on session state."""
        self.session_active = active
        
        for widget in self.session_sensitive_widgets:
            try:
                # Get the widget's desired state based on whether it's a session control
                is_control = getattr(widget, 'is_session_control', False)
                if is_control:
                    # Session controls are enabled during session, disabled otherwise
                    state = "normal" if active else "disabled"
                else:
                    # Setup controls are disabled during session, enabled otherwise
                    state = "disabled" if active else "normal"
                
                # Apply the appropriate state based on widget type
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="disabled" if state == "disabled" else "readonly")
                else:
                    widget.configure(state=state)
            except tk.TclError as e:
                print(f"Warning: Could not configure widget {widget}: {e}")

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
        # Re-enable all widgets after cleanup
        self._set_session_state(False)
    
    def kill_process_tree(self, pid):
        """Kill a process and all its children."""
        if not PSUTIL_AVAILABLE:
            self.log_to_console("psutil not available - using basic termination")
            return
            
        try:
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
        # Prevent stopping if no session is active
        if not self.session_active:
            messagebox.showwarning("No Active Session", "There is no active session to stop.")
            return
        
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to stop the current session?"):
            self.log_to_console("\nShutting down...")
            
            # Log final session state before shutdown
            try:
                if hasattr(self, 'log_file'):
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")
                    current_level = self.current_level.get()
                    batch_id = self.batch_id.get()
                    reward_count = self.reward_count.get()
                    
                    with open(self.log_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([date_str, time_str, f"{current_level} (Session End - {reward_count} rewards)", batch_id])
                    self.log_to_console(f"Final session state logged: Level={current_level}, Rewards={reward_count}")
            except Exception as e:
                self.log_to_console(f"Error logging final session state: {str(e)}")
            
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
    
    def _update_reward_display(self, count, threshold_reached=False):
        """Callback function to update the reward display when new rewards are received."""
        # Update the counter display (convert to string once)
        start_time = time.time()
        self.reward_count.set(str(count))
        update_time = time.time() - start_time
        
        if update_time > 0.01:  # If update takes more than 10ms, log it
            print(f"WARNING: Reward display update took {update_time*1000:.1f}ms")
        
        # Only log significant events, not every reward
        # This prevents console spam and improves GUI responsiveness
        if threshold_reached:
            self.log_to_console(f"Reward threshold reached! Total rewards: {count}")
            self.log_to_console("Advancing to next level...")
            # Schedule the level change on the main thread with after() instead of after_idle()
            # after(1) is faster than after_idle() because it doesn't wait for idle state
            self.root.after(1, self.change_to_next_level)
        elif count % 10 == 0:  # Only log every 10th reward to reduce console spam
            self.log_to_console(f"Rewards: {count}")
    
    def start_session(self):
        # Prevent starting multiple sessions
        if self.session_active:
            messagebox.showerror("Error", "A session is already running")
            return
            
        # Validate inputs
        if not all([self.animal_name.get(), self.batch_id.get(), self.teensy_port.get()]):
            messagebox.showerror("Error", "Please fill in all required fields")
            return
        
        # Determine the starting level from the progress report
        animal_name = self.animal_name.get()
        report_file = f"{animal_name}_log.csv"
        starting_level, starting_reward_count = self.get_starting_level_from_report(report_file)
        
        # Get file paths
        level_file = os.path.join(os.getcwd(), 'Levels', starting_level)
        phase_file = os.path.join(os.getcwd(), 'stopping_control.py')
        
        # Log the run
        self.log_run(animal_name, level_file, self.batch_id.get())
        
        # Convert level file path to relative with forward slashes
        level_file_path = level_file.replace(os.getcwd() + os.sep, '').replace('\\', '/')
        
        try:
            # Start TCP server
            self.tcp_server = TCPDataServer()
            server_port = self.tcp_server.start_server()
            
            # Override the callback scheduler to use Tkinter's thread-safe after() method
            def thread_safe_schedule(count, threshold_reached):
                # This runs in the TCP thread, schedule the actual callback on the main thread
                self.root.after(0, lambda: self._update_reward_display(count, threshold_reached))
            
            self.tcp_server._schedule_callback = thread_safe_schedule
            
            # Set up reward counting callback
            self.tcp_server.set_reward_callback(self._update_reward_display)
            
            # Set the initial level index and threshold
            initial_level = starting_level
            if not self.tcp_server.set_initial_level(initial_level):
                raise Exception(f"Invalid starting level: {initial_level}")
            
            # Set initial reward threshold
            try:
                self.tcp_server.current_reward_threshold = self.tcp_server._get_level_reward_threshold(initial_level)
                self.log_to_console(f"Initial reward threshold set to: {self.tcp_server.current_reward_threshold}")
            except Exception as e:
                self.log_to_console(f"Error setting initial reward threshold: {e}")
                raise
            
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
            self.current_level.set(starting_level)
            
            # Initialize reward counting for the first level - use the reward count from progress report
            self.tcp_server.reset_reward_count()
            # Manually set the reward count to the value from the progress report
            with self.tcp_server._reward_lock:
                self.tcp_server.reward_count = starting_reward_count
            # Update the display
            self.reward_count.set(str(starting_reward_count))
            
            # Log success
            self.log_to_console(f"Started session with:")
            self.log_to_console(f"Animal: {self.animal_name.get()}")
            self.log_to_console(f"Level: {starting_level}")
            self.log_to_console(f"Starting Reward Count: {starting_reward_count}")
            self.log_to_console(f"Batch ID: {self.batch_id.get()}")
            self.log_to_console(f"Teensy Port: {self.teensy_port.get()}")
            self.log_to_console(f"TCP server started on port {server_port}")
            
            # Disable setup controls while session is running
            self._set_session_state(True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start session: {str(e)}")
            self.cleanup_resources()  # This will also re-enable controls

def main():
    root = tk.Tk()
    app = KaufmanGUI(root)
    
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