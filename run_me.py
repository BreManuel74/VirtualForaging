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

def list_files(folder, ext=".py"):
    return [f for f in os.listdir(folder) if f.endswith(ext)]

def list_dirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

def select_file(folder_name, ext=".py"):
    folder_path = os.path.join(os.getcwd(), folder_name)
    files = list_files(folder_path, ext)
    print(f"Files in {folder_name}:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}: {file}")
    choice = int(input(f"Select a file from {folder_name} (1-{len(files)}): ")) - 1
    return os.path.join(folder_path, files[choice])

def select_dir(base_folder):
    while True:
        dirs = list_dirs(base_folder)
        print(f"\nCurrent directory: {base_folder}")
        if not dirs:
            #print(f"No directories found in {base_folder}.")
            return base_folder
        print("0: Select this directory")
        for idx, d in enumerate(dirs):
            print(f"{idx + 1}: {d}")
        choice = int(input(f"Select a directory (1-{len(dirs)}), or 0 to use this directory: "))
        if choice == 0:
            return base_folder
        base_folder = os.path.join(base_folder, dirs[choice - 1])

def make_relative_forward_slash(path):
    rel_path = os.path.relpath(path, os.getcwd())
    return rel_path.replace("\\", "/")

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
        
    def _get_available_levels(self):
        """Get list of available level files."""
        try:
            levels_path = os.path.join(os.getcwd(), self.levels_folder)
            return [f for f in os.listdir(levels_path) if f.endswith('.json')]
        except:
            return []
    
    def list_available_levels(self):
        """Return formatted list of available levels."""
        if not self.available_levels:
            return "No level files found."
        
        level_list = "Available levels:\n"
        for idx, level in enumerate(self.available_levels, 1):
            level_list += f"{idx}: {level}\n"
        return level_list.strip()
        
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
        return self.send_data(command)
    
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

def _interactive_command_interface(tcp_server):
    """Interactive interface for sending commands to the game with improved error handling."""
    try:
        print("\n=== Interactive Command Interface ===")
        print("Type commands to send to the game:")
        print("Available commands:")
        print("  'change_level' - Change to a different level file")
        print("  'list_levels' - Show available level files")
        print("  'quit' or 'exit' - Stop sending commands")
        print()
        
        while tcp_server.running:
            try:
                command = input("Command: ").strip().lower()
                
                if command in ['quit', 'exit']:
                    break
                elif command == 'list_levels':
                    print(tcp_server.list_available_levels())
                elif command == 'change_level':
                    print(tcp_server.list_available_levels())
                    try:
                        choice = int(input("Select level number: ")) - 1
                        if 0 <= choice < len(tcp_server.available_levels):
                            selected_level = tcp_server.available_levels[choice]
                            success = tcp_server.send_level_change(selected_level)
                            if success:
                                print(f"Sent level change to: {selected_level}")
                            else:
                                print(f"Failed to send level change command")
                        else:
                            print("Invalid selection")
                    except (ValueError, IndexError):
                        print("Invalid input. Please enter a number.")
                elif command:
                    success = tcp_server.send_data(command)
                    if success:
                        print(f"Sent: {command}")
                    else:
                        print(f"Failed to send: {command}")
            except (EOFError, KeyboardInterrupt):
                print("\nCommand interface interrupted")
                break
            except Exception as e:
                print(f"Error in command interface: {e}")
                break
    except Exception as e:
        print(f"Fatal error in command interface: {e}")
    finally:
        print("Command interface stopped.")

def change_level_by_name(tcp_server, level_name):
    """
    Programmatically change to a specific level file.
    
    Args:
        tcp_server: The TCPDataServer instance
        level_name: Name of the level file (with or without .json extension)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not level_name.endswith('.json'):
        level_name += '.json'
    
    if level_name in tcp_server.available_levels:
        return tcp_server.send_level_change(level_name)
    else:
        print(f"Level '{level_name}' not found in available levels")
        return False

def log_run(animal_name, level_file, phase_file, batch_id):
    log_dir = "Progress_Reports"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    log_file = os.path.join(log_dir, f"{animal_name}_log.csv")
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    # Only log the file names, not full paths
    level_file_name = os.path.basename(level_file)
    phase_file_name = os.path.basename(phase_file)
    # Write header if file does not exist
    write_header = not os.path.exists(log_file)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Date", "Time", "Level File", "Phase File", "Batch ID"])
        writer.writerow([date_str, time_str, level_file_name, phase_file_name, batch_id])

def run_phase_with_level(phase_file_path, level_file_path, output_dir=None, batch_id=None, teensy_port=None):
    level_file_path = make_relative_forward_slash(level_file_path)
    print(f"About to run: {phase_file_path} with config: {level_file_path}")
    
    tcp_server = None
    process = None
    command_thread = None
    
    def signal_handler(signum, frame):
        """Handle system signals for graceful shutdown."""
        print(f"\nReceived signal {signum}, shutting down...")
        cleanup_resources()
        sys.exit(0)
    
    def cleanup_resources():
        """Clean up all resources."""
        print("Cleaning up resources...")
        if tcp_server:
            tcp_server.stop_server()
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start TCP server
        tcp_server = TCPDataServer()
        server_port = tcp_server.start_server()
        
        # Set up environment variables
        env = os.environ.copy()
        env["LEVEL_CONFIG_PATH"] = level_file_path
        env["TCP_SERVER_PORT"] = str(server_port)
        if output_dir:
            env["OUTPUT_DIR"] = output_dir
        if batch_id is not None:
            env["BATCH_ID"] = str(batch_id)
        if teensy_port:
            env["TEENSY_PORT"] = teensy_port
        
        # Start the Panda3D game as a subprocess
        process = subprocess.Popen([sys.executable, phase_file_path], env=env)
        
        # Wait a moment for the game to start and connect
        time.sleep(2)
        
        # Print available commands and levels
        print("TCP server ready. You can now send data to the game.")
        print("Use the interactive interface to:")
        print("- Change level files dynamically ('change_level')")
        print("- List available levels ('list_levels')")
        print()
        
        # Start interactive command interface in a separate thread
        command_thread = threading.Thread(
            target=_interactive_command_interface, 
            args=(tcp_server,), 
            daemon=True, 
            name="CommandInterface"
        )
        command_thread.start()
        
        # Wait for the subprocess to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        cleanup_resources()
        print("Application stopped")
    
    #print("Phase file loaded!")

if __name__ == "__main__":
    # Example usage for level changing:
    # 1. Start the script normally
    # 2. In the interactive interface, type:
    #    - 'list_levels' to see all available levels
    #    - 'change_level' to select a new level
    #    - Any game command like 'reward', 'puff', etc.
    # 3. Or programmatically: change_level_by_name(tcp_server, "final_level_1")
    
    animal_name = input("Enter animal name: ")
    level_file = select_file('Levels', '.json')
    phase_file = select_file('Phases', '.py')
    output_dir = select_dir(os.getcwd())
    batch_id = input("Enter batch ID number: ")
    teensy_port = input("Enter Teensy port (e.g., COM3): ")
    #print(f"Running {phase_file} with level config {level_file}, OUTPUT_DIR {output_dir}, and BATCH_ID {batch_id}...")
    log_run(animal_name, level_file, phase_file, batch_id)
    run_phase_with_level(phase_file, level_file, output_dir, batch_id, teensy_port)
